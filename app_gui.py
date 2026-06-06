# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cloud Task Scheduling GUI", layout="wide")

st.title("☁️ Cloud Task Scheduling Optimizer")
st.markdown("Aplikasi GUI untuk membandingkan performa penjadwalan (Modified Branch and Bound) dengan berbagai skema **Sample Size** dan **Jumlah VM**.")

# ---------------------------------------------------------
# 1. FUNGSI LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"File {filepath} tidak ditemukan!")
        return None
        
    df = pd.read_csv(filepath)
    required_columns = [
        "Task_ID", "Task_Length_MIPS", "Task_Deadline", "Data_Upload_Size_MB", 
        "Data_Download_Size_MB", "VM_ID", "VM_MIPS", "VM_Memory_GB", 
        "VM_Bandwidth_MBps", "Task_Priority", "Execution_Time_S", 
        "Energy_Consumption_J", "Execution_Cost_$"
    ]
    df_clean = df.copy()
    for col in required_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            
    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean[
        (df_clean["Task_Length_MIPS"] > 0) & (df_clean["Task_Deadline"] > 0) &
        (df_clean["VM_MIPS"] > 0) & (df_clean["VM_Bandwidth_MBps"] > 0) &
        (df_clean["Execution_Time_S"] > 0) & (df_clean["Energy_Consumption_J"] > 0) &
        (df_clean["Execution_Cost_$"] > 0)
    ].copy()
    
    EPS = 1e-9
    df_clean["power_rate_jps"] = df_clean["Energy_Consumption_J"] / (df_clean["Execution_Time_S"] + EPS)
    df_clean["cost_rate_per_s"] = df_clean["Execution_Cost_$"] / (df_clean["Execution_Time_S"] + EPS)
    
    return df_clean

# ---------------------------------------------------------
# 2. FUNGSI OPTIMASI (Diambil dari Notebook)
# ---------------------------------------------------------
def run_optimization(df_clean, sample_size, max_vm, top_k_vm, node_limit, time_limit_sec):
    EPS = 1e-9
    WEIGHTS = {"energy": 0.35, "cost": 0.35, "penalty": 0.10, "time": 0.20}
    
    # Siapkan VM Table
    vm_table = (
        df_clean.groupby("VM_ID", as_index=False)
        .agg(
            VM_MIPS=("VM_MIPS", "median"),
            VM_Memory_GB=("VM_Memory_GB", "median"),
            VM_Bandwidth_MBps=("VM_Bandwidth_MBps", "median"),
            power_rate_jps=("power_rate_jps", "median"),
            cost_rate_per_s=("cost_rate_per_s", "median")
        )
        .sort_values("VM_MIPS", ascending=False)
        .head(max_vm).reset_index(drop=True)
    )

    # Siapkan Task Sample
    task_sample = df_clean.sample(n=min(sample_size, len(df_clean)), random_state=42).reset_index(drop=True)

    n_tasks = len(task_sample)
    m_vms = len(vm_table)

    def normalize_matrix(matrix):
        matrix = np.asarray(matrix, dtype=float)
        min_value = np.nanmin(matrix)
        max_value = np.nanmax(matrix)
        if abs(max_value - min_value) < EPS: return np.zeros_like(matrix)
        return (matrix - min_value) / (max_value - min_value + EPS)

    def normalize_priority(series):
        values = pd.to_numeric(series, errors="coerce").fillna(1).to_numpy(dtype=float)
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        if abs(max_value - min_value) < EPS: return np.ones_like(values)
        return 1 + ((values - min_value) / (max_value - min_value + EPS))

    task_length = task_sample["Task_Length_MIPS"].to_numpy(dtype=float)
    deadline = task_sample["Task_Deadline"].to_numpy(dtype=float)
    upload = task_sample["Data_Upload_Size_MB"].to_numpy(dtype=float)
    download = task_sample["Data_Download_Size_MB"].to_numpy(dtype=float)
    priority_weight = normalize_priority(task_sample["Task_Priority"])

    vm_mips = vm_table["VM_MIPS"].to_numpy(dtype=float)
    vm_bandwidth = vm_table["VM_Bandwidth_MBps"].to_numpy(dtype=float)
    power_rate = vm_table["power_rate_jps"].to_numpy(dtype=float)
    cost_rate = vm_table["cost_rate_per_s"].to_numpy(dtype=float)

    compute_time = task_length[:, None] / (vm_mips[None, :] + EPS)
    transfer_time = (upload[:, None] + download[:, None]) / (vm_bandwidth[None, :] + EPS)
    runtime_matrix = compute_time + transfer_time

    energy_matrix = runtime_matrix * power_rate[None, :]
    cost_matrix = runtime_matrix * cost_rate[None, :]

    norm_energy = normalize_matrix(energy_matrix)
    norm_cost = normalize_matrix(cost_matrix)

    penalty_reference = max(float(np.nanmax(deadline)), float(np.nanmax(runtime_matrix)) * n_tasks, 1.0)
    time_reference = max(float(np.nanmax(runtime_matrix)) * n_tasks, float(np.nanmax(deadline)), 1.0)

    static_lower_score_matrix = (
        WEIGHTS["energy"] * norm_energy +
        WEIGHTS["cost"] * norm_cost +
        WEIGHTS["time"] * (runtime_matrix / time_reference)
    )

    def evaluate_assignment(assignment, start_time_arr, finish_time_arr):
        assignment = np.asarray(assignment, dtype=int)
        total_energy, total_cost, total_penalty, total_time_component, total_score = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i in range(n_tasks):
            vm_idx = assignment[i]
            finish = finish_time_arr[i]
            penalty = max(0.0, finish - deadline[i]) * priority_weight[i]
            time_component = finish / time_reference
            score = (
                WEIGHTS["energy"] * norm_energy[i, vm_idx] +
                WEIGHTS["cost"] * norm_cost[i, vm_idx] +
                WEIGHTS["penalty"] * (penalty / penalty_reference) +
                WEIGHTS["time"] * time_component
            )
            total_energy += energy_matrix[i, vm_idx]
            total_cost += cost_matrix[i, vm_idx]
            total_penalty += penalty
            total_time_component += time_component
            total_score += score
            
        return {
            "assignment": assignment.copy(),
            "total_energy": total_energy,
            "total_cost": total_cost,
            "total_penalty": total_penalty,
            "makespan": float(np.max(finish_time_arr)),
            "objective_score": total_score,
        }

    def greedy_schedule(order, candidate_vms=None, mode="time_aware"):
        vm_load = np.zeros(m_vms, dtype=float)
        assignment = np.full(n_tasks, -1, dtype=int)
        start_time_arr = np.zeros(n_tasks, dtype=float)
        finish_time_arr = np.zeros(n_tasks, dtype=float)

        for task_idx in order:
            candidates = candidate_vms[task_idx] if candidate_vms else range(m_vms)
            best_vm, best_value = None, float("inf")

            for vm_idx in candidates:
                start = vm_load[vm_idx]
                finish = start + runtime_matrix[task_idx, vm_idx]
                penalty = max(0.0, finish - deadline[task_idx]) * priority_weight[task_idx]
                
                value = (
                    WEIGHTS["energy"] * norm_energy[task_idx, vm_idx] +
                    WEIGHTS["cost"] * norm_cost[task_idx, vm_idx] +
                    WEIGHTS["penalty"] * (penalty / penalty_reference) +
                    WEIGHTS["time"] * (finish / time_reference)
                )

                if value < best_value:
                    best_value = value
                    best_vm = vm_idx

            start_time_arr[task_idx] = vm_load[best_vm]
            finish_time_arr[task_idx] = vm_load[best_vm] + runtime_matrix[task_idx, best_vm]
            vm_load[best_vm] = finish_time_arr[task_idx]
            assignment[task_idx] = best_vm

        return evaluate_assignment(assignment, start_time_arr, finish_time_arr), start_time_arr, finish_time_arr

    # Task Ordering & Candidates
    top_candidate_vms = {
        i: np.argsort(static_lower_score_matrix[i])[:min(top_k_vm, m_vms)].tolist()
        for i in range(n_tasks)
    }
    modified_order = sorted(
        range(n_tasks),
        key=lambda i: (
            deadline[i] / (np.min(runtime_matrix[i]) + EPS),
            np.min(runtime_matrix[i]),
            np.min(static_lower_score_matrix[i])
        )
    )

    # Initial Greedy Solution
    greedy_result, g_start, g_finish = greedy_schedule(modified_order, top_candidate_vms, mode="time_aware")
    
    # Branch and Bound
    best_score = greedy_result["objective_score"]
    best_assignment = greedy_result["assignment"].copy()
    best_start = g_start.copy()
    best_finish = g_finish.copy()

    suffix_lower_bound = np.zeros(n_tasks + 1, dtype=float)
    for pos in range(n_tasks - 1, -1, -1):
        task_idx = modified_order[pos]
        suffix_lower_bound[pos] = suffix_lower_bound[pos + 1] + np.min(static_lower_score_matrix[task_idx, top_candidate_vms[task_idx]])

    vm_load = np.zeros(m_vms, dtype=float)
    assignment = np.full(n_tasks, -1, dtype=int)
    start_time_arr = np.zeros(n_tasks, dtype=float)
    finish_time_arr = np.zeros(n_tasks, dtype=float)

    stats = {"nodes_visited": 0, "pruned_nodes": 0, "hit_limit": False}
    search_start = time.time()

    def dfs(pos, current_score):
        nonlocal best_score, best_assignment, best_start, best_finish

        if stats["hit_limit"]: return
        if stats["nodes_visited"] >= node_limit or (time.time() - search_start) >= time_limit_sec:
            stats["hit_limit"] = True
            return

        stats["nodes_visited"] += 1

        if current_score + suffix_lower_bound[pos] >= best_score:
            stats["pruned_nodes"] += 1
            return

        if pos == n_tasks:
            if current_score < best_score:
                best_score = current_score
                best_assignment = assignment.copy()
                best_start = start_time_arr.copy()
                best_finish = finish_time_arr.copy()
            return

        task_idx = modified_order[pos]
        ordered_candidates = sorted(
            top_candidate_vms[task_idx],
            key=lambda vm_idx: (
                WEIGHTS["energy"] * norm_energy[task_idx, vm_idx] +
                WEIGHTS["cost"] * norm_cost[task_idx, vm_idx] +
                WEIGHTS["time"] * ((vm_load[vm_idx] + runtime_matrix[task_idx, vm_idx]) / time_reference)
            )
        )

        for vm_idx in ordered_candidates:
            start = vm_load[vm_idx]
            finish = start + runtime_matrix[task_idx, vm_idx]
            penalty = max(0.0, finish - deadline[task_idx]) * priority_weight[task_idx]

            incremental_score = (
                WEIGHTS["energy"] * norm_energy[task_idx, vm_idx] +
                WEIGHTS["cost"] * norm_cost[task_idx, vm_idx] +
                WEIGHTS["penalty"] * (penalty / penalty_reference) +
                WEIGHTS["time"] * (finish / time_reference)
            )

            new_score = current_score + incremental_score

            if new_score + suffix_lower_bound[pos + 1] >= best_score:
                stats["pruned_nodes"] += 1
                continue

            previous_load = vm_load[vm_idx]
            assignment[task_idx] = vm_idx
            start_time_arr[task_idx] = start
            finish_time_arr[task_idx] = finish
            vm_load[vm_idx] = finish

            dfs(pos + 1, new_score)

            vm_load[vm_idx] = previous_load
            assignment[task_idx] = -1

    dfs(0, 0.0)

    final_result = evaluate_assignment(best_assignment, best_start, best_finish)
    
    return {
        "Skema": f"N={sample_size}, VM={max_vm}",
        "Sample_Size": sample_size,
        "Max_VM": max_vm,
        "Makespan_S": round(final_result["makespan"], 2),
        "Energy_J": round(final_result["total_energy"], 4),
        "Cost_$": round(final_result["total_cost"], 2),
        "Objective_Score": round(final_result["objective_score"], 2),
        "Runtime_S": round(time.time() - search_start, 2),
        "Nodes_Visited": stats["nodes_visited"],
        "Hit_Limit": stats["hit_limit"]
    }

# ---------------------------------------------------------
# 3. INTERFACE GUI STREAMLIT
# ---------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Skema")
st.sidebar.markdown("Masukkan nilai yang dipisahkan dengan koma (misal: `30, 50, 100`) untuk membandingkan berbagai kombinasi.")

sample_sizes_str = st.sidebar.text_input("Daftar Sample Size (N)", "30, 50")
max_vms_str = st.sidebar.text_input("Daftar Jumlah VM", "4, 8")

st.sidebar.subheader("Parameter Branch & Bound")
top_k = st.sidebar.slider("Top K Kandidat VM", 1, 10, 5)
node_limit = st.sidebar.number_input("Node Limit", min_value=1000, value=100000, step=10000)
time_limit = st.sidebar.number_input("Time Limit (Detik)", min_value=1, value=30, step=5)

data_path = "Distributed_Task_Scheduling (1).csv"
df_clean = load_and_clean_data(data_path)

if st.sidebar.button("🚀 Jalankan Perbandingan", type="primary"):
    if df_clean is None:
        st.error("Dataset tidak valid. Cek path file.")
    else:
        try:
            # Parse input
            sample_sizes = [int(s.strip()) for s in sample_sizes_str.split(",")]
            max_vms = [int(v.strip()) for v in max_vms_str.split(",")]
            
            total_experiments = len(sample_sizes) * len(max_vms)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            counter = 0
            for s in sample_sizes:
                for v in max_vms:
                    status_text.text(f"Menjalankan skema: Sample Size = {s}, VM = {v} ...")
                    
                    res = run_optimization(df_clean, s, v, top_k, node_limit, time_limit)
                    results.append(res)
                    
                    counter += 1
                    progress_bar.progress(counter / total_experiments)
                    
            status_text.text("Semua skema selesai dijalankan!")
            st.success("Eksperimen selesai!")
            
            # --- TAMPILKAN HASIL ---
            df_results = pd.DataFrame(results)
            
            st.subheader("📊 Tabel Hasil Perbandingan")
            st.dataframe(
                df_results.style
                .format({
                    "Makespan_S": "{:.2f}",
                    "Energy_J": "{:.4f}",
                    "Cost_$": "{:.2f}",
                    "Objective_Score": "{:.2f}",
                    "Runtime_S": "{:.2f}"
                })
                .highlight_min(subset=['Objective_Score', 'Makespan_S', 'Energy_J', 'Cost_$'], color='lightgreen')
            )
            
            st.subheader("📈 Grafik Perbandingan")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Perbandingan Objective Score** (Semakin kecil semakin baik)")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.bar(df_results["Skema"], df_results["Objective_Score"], color='skyblue')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig1)

                st.markdown("**Perbandingan Total Cost ($)**")
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                ax3.bar(df_results["Skema"], df_results["Cost_$"], color='salmon')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig3)
                
            with col2:
                st.markdown("**Perbandingan Makespan (Detik)**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.bar(df_results["Skema"], df_results["Makespan_S"], color='lightgreen')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig2)
                
                st.markdown("**Perbandingan Konsumsi Energi (Joule)**")
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                ax4.bar(df_results["Skema"], df_results["Energy_J"], color='orange')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig4)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan atur parameter di sidebar dan klik 'Jalankan Perbandingan'.")
