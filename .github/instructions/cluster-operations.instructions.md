---
applyTo: "cpp/simulation/cluster/**"
---

# Cluster Operations - Copilot Instructions

This document describes operations on the Nibi cluster (nibi.alliancecan.ca) for running cell simulations at scale.

## Overview

- **Cluster**: Nibi (Compute Canada / Alliance Canada)
- **Account**: `rrg-mkarttu-ab`
- **Username**: `ssilber`
- **Scratch storage**: `/scratch/ssilber/`
- **Home directory**: `~/cell_simulation/` (code), `~/cell_sim_logs/` (logs)

## SSH Connection Management

### The MFA Problem
Nibi requires Duo MFA for every SSH connection. To avoid repeated authentication, use SSH ControlMaster to maintain a persistent connection.

### Establish Persistent Connection (Do This First)
```powershell
# From Windows PowerShell, via WSL
# This will prompt for MFA ONCE, then persist for 4 hours

wsl bash -c "mkdir -p ~/.ssh/sockets"
wsl ssh -M -S ~/.ssh/sockets/nibi -o ControlPersist=4h -o ServerAliveInterval=60 -o ServerAliveCountMax=3 ssilber@nibi.alliancecan.ca "echo 'Connection established'; hostname"
```

**What the flags do:**
- `-M`: Creates a master connection
- `-S ~/.ssh/sockets/nibi`: Socket file for multiplexing
- `-o ControlPersist=4h`: Keep connection alive for 4 hours
- `-o ServerAliveInterval=60`: Send keepalive every 60 seconds
- `-o ServerAliveCountMax=3`: Disconnect after 3 missed keepalives

### Using the Persistent Connection

Once the master connection is established, all subsequent commands use it automatically:

```powershell
# Run commands (no MFA needed)
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "command here"

# File transfer (no MFA needed)
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" local_file ssilber@nibi.alliancecan.ca:/scratch/ssilber/

# Download files
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" ssilber@nibi.alliancecan.ca:/scratch/ssilber/results.tar ./
```

### Check Connection Status
```powershell
# Check if socket exists and connection is alive
wsl ssh -S ~/.ssh/sockets/nibi -O check ssilber@nibi.alliancecan.ca
```

### Refresh/Reconnect When Expired
```powershell
# If connection died, clear socket and reconnect
wsl bash -c "rm -f ~/.ssh/sockets/nibi"
wsl ssh -M -S ~/.ssh/sockets/nibi -o ControlPersist=4h -o ServerAliveInterval=60 ssilber@nibi.alliancecan.ca "hostname"
```

### Recommended: Add to ~/.ssh/config (in WSL)
```bash
# Run: wsl nano ~/.ssh/config
# Add these lines:

Host nibi
    HostName nibi.alliancecan.ca
    User ssilber
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 4h
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Then you can simply use:
```powershell
wsl ssh nibi "command"
wsl scp file nibi:/scratch/ssilber/
```

## Key Cluster Paths

| Path | Purpose |
|------|---------|
| `~/cell_simulation/` | Main codebase (home directory) |
| `~/cell_simulation/build/bin/cell_sim` | Compiled executable |
| `~/cell_simulation/cluster/` | Job scripts |
| `~/cell_sim_logs/` | Job output logs (.out, .err) |
| `~/cell_sim_logs/submitted_jobs.txt` | **Submission log** (job IDs) |
| `~/cell_sim_logs/job_status.txt` | Job status history |
| `/scratch/ssilber/cell_sim_results/` | Simulation output |
| `/scratch/ssilber/jamming_study/` | Active production runs |

## Building on Cluster

### Full Build (First Time)
```bash
# SSH into cluster first
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca

# On cluster:
module load cuda/12.2 cmake/3.27 gcc/12.3
cd ~/cell_simulation
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
make -j8
```

### Quick Rebuild After Code Changes
```bash
cd ~/cell_simulation/build
make -j8
```

### Build Script (Automated)
```bash
# Run the build script
cd ~/cell_simulation/cluster
./build_on_cluster.sh
```

### Uploading Code Changes
```powershell
# From local Windows machine
cd C:\Users\stevensilber\source\repos\data_processing\cpp\simulation

# Create tarball of source
tar -cvf simulation_src.tar src include cluster CMakeLists.txt

# Upload via persistent connection
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" simulation_src.tar ssilber@nibi.alliancecan.ca:~/

# Extract and rebuild on cluster
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cd ~ && tar -xf simulation_src.tar -C cell_simulation && cd cell_simulation/build && make -j8"
```

## Submitting Jobs

### Using the Submit Script (Recommended)

The `submit.sh` script handles job creation, logging, and parameter tracking:

```bash
# From cluster: ~/cell_simulation/cluster/
./submit.sh quick_test           # 8 cells, t=10, 30 min
./submit.sh small                # 32 cells, t=100, 6 hr
./submit.sh medium               # 64 cells, t=100, 12 hr
./submit.sh large                # 128 cells, t=100, R=32, 24 hr
./submit.sh xl                   # 200 cells, t=100, R=32, 48 hr

# Custom cell count
./submit.sh 48                   # 48 cells with defaults
./submit.sh 48 -r 32 -t 200      # 48 cells, R=32, t=200
```

### From Windows (via persistent connection)
```powershell
# Submit a job remotely
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cd ~/cell_simulation/cluster && ./submit.sh medium"
```

### Manual SBATCH Submission
```bash
# For custom jobs, create a script or use job_3d.sh
sbatch job_3d.sh 64 49 100 my_custom_run
# Args: n_cells radius t_end output_name
```

### Job Presets Reference

| Preset | Cells | Radius | Time | Wall Time | Use Case |
|--------|-------|--------|------|-----------|----------|
| `quick_test` | 8 | 49 | 10 | 30 min | Verify build works |
| `small` | 32 | 49 | 100 | 6 hr | Development testing |
| `medium` | 64 | 49 | 100 | 12 hr | Standard runs |
| `large` | 128 | 32 | 100 | 24 hr | High cell count |
| `xl` | 200 | 32 | 100 | 48 hr | Production scale |

### SLURM Resource Configuration

Standard job resources (in job scripts):
```bash
#SBATCH --account=rrg-mkarttu-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1        # 1 GPU (A100)
#SBATCH --mem=32G
```

For larger jobs:
```bash
#SBATCH --mem=64G                # More memory for large domains
#SBATCH --time=48:00:00          # Up to 48 hours
```

## Submission Log System

### How It Works

Every job submission is logged to `~/cell_sim_logs/submitted_jobs.txt`:
```
<job_id> <run_name>
```

Example contents:
```
5516400 quick_test_n8_r49_20251208_181408
5517005 quick_test_n8_r49_20251208_182333
5517273 medium_64c_n64_r49_20251208_182711
```

### Viewing the Submission Log
```powershell
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cat ~/cell_sim_logs/submitted_jobs.txt"
```

### Status History

Job lifecycle events are logged to `~/cell_sim_logs/job_status.txt`:
```
QUEUED: 5516400 quick_test_n8_r49_20251208_181408 submitted Mon Dec  8 18:14:08 EST 2025
RUNNING: 5516400 quick_test_n8_r49_20251208_181408 started Mon Dec  8 18:15:01 EST 2025
COMPLETED: 5516400 quick_test_n8_r49_20251208_181408 finished Mon Dec  8 18:15:01 EST 2025
```

### Extracting Job Parameters

Each completed job saves a `run_info.json` in its output directory:
```json
{
  "job_id": "5517273",
  "run_name": "medium_64c_n64_r49_20251208_182711",
  "n_cells": 64,
  "radius": 49,
  "t_end": 100,
  "status": "completed",
  "completed_at": "2025-12-08T18:47:23-05:00"
}
```

### Query All Job Parameters
```powershell
# List all run_info.json files and their contents
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "for d in /scratch/ssilber/cell_sim_results/*/; do [ -f \"\$d/run_info.json\" ] && echo '=== '\$d && cat \"\$d/run_info.json\"; done"
```

### Get Parameters for Specific Job
```powershell
# By job ID
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "grep 5517273 ~/cell_sim_logs/submitted_jobs.txt && find /scratch/ssilber/cell_sim_results -name run_info.json -exec grep -l 5517273 {} \; -exec cat {} \;"
```

## Monitoring Jobs

### Check Queue Status
```powershell
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "squeue -u ssilber"
```

### View Job Output (Live)
```powershell
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "tail -f ~/cell_sim_logs/*.out"
```

### Check Recent Completed Jobs
```powershell
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "sacct -u ssilber --starttime=\$(date -d '24 hours ago' +%Y-%m-%d) --format=JobID,JobName%30,State,Elapsed"
```

### Full Status Report
```powershell
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cd ~/cell_simulation/cluster && ./status.sh"
```

## Downloading Results

### Download Specific Run
```powershell
wsl scp -r -o "ControlPath=~/.ssh/sockets/nibi" ssilber@nibi.alliancecan.ca:/scratch/ssilber/cell_sim_results/medium_64c_n64_r49_20251208_182711 ./cluster_results/
```

### Download Latest Run
```powershell
# Find latest
$latest = wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "ls -t /scratch/ssilber/cell_sim_results | head -1"
Write-Host "Downloading: $latest"
wsl scp -r -o "ControlPath=~/.ssh/sockets/nibi" "ssilber@nibi.alliancecan.ca:/scratch/ssilber/cell_sim_results/$latest" ./cluster_results/
```

### Download Just Checkpoints/Trajectories (Skip VTK)
```powershell
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cd /scratch/ssilber/cell_sim_results/my_run && tar -cvf ~/results_lite.tar checkpoint.bin trajectory.txt run_info.json *.csv"
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" ssilber@nibi.alliancecan.ca:~/results_lite.tar ./
```

## Using the PowerShell Management Script

The `nibi.ps1` script provides a convenient interface from Windows:

```powershell
cd C:\Users\stevensilber\source\repos\data_processing\cpp\simulation\cluster

# Check status
.\nibi.ps1 status

# Submit jobs
.\nibi.ps1 submit quick_test
.\nibi.ps1 submit medium
.\nibi.ps1 submit 48 -r 32 -t 200

# Sync code
.\nibi.ps1 sync

# Download results
.\nibi.ps1 download              # Latest
.\nibi.ps1 download run_name     # Specific run

# View logs
.\nibi.ps1 logs                  # Recent logs
.\nibi.ps1 logs 5517273          # Specific job
```

**Note**: The script uses direct `ssh` commands, not the socket. For socket-based access, use the WSL commands directly.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Connection refused" | Socket expired; re-establish master connection with MFA |
| "Permission denied" | Check SSH key is loaded: `wsl ssh-add -l` |
| Job stuck in PENDING | Check account quota: `sshare -u ssilber` |
| Job failed immediately | Check error log: `cat ~/cell_sim_logs/<run>.err` |
| Out of disk space | Clean old results: `rm -rf /scratch/ssilber/cell_sim_results/old_run` |
| Module not found | Run: `module spider cuda` to find available versions |
| Build fails | Ensure modules loaded: `module load cuda/12.2 cmake/3.27 gcc/12.3` |

## Quick Reference Commands

```powershell
# === Connection ===
# Establish (with MFA)
wsl ssh -M -S ~/.ssh/sockets/nibi -o ControlPersist=4h ssilber@nibi.alliancecan.ca "hostname"

# Test connection
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "echo ok"

# === Jobs ===
# Submit
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cd ~/cell_simulation/cluster && ./submit.sh medium"

# Status
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "squeue -u ssilber"

# Cancel job
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "scancel <job_id>"

# === Logs ===
# Submission log
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cat ~/cell_sim_logs/submitted_jobs.txt"

# Job status history
wsl ssh -S ~/.ssh/sockets/nibi ssilber@nibi.alliancecan.ca "cat ~/cell_sim_logs/job_status.txt"

# === Files ===
# Upload
wsl scp -o "ControlPath=~/.ssh/sockets/nibi" file.tar ssilber@nibi.alliancecan.ca:/scratch/ssilber/

# Download
wsl scp -r -o "ControlPath=~/.ssh/sockets/nibi" ssilber@nibi.alliancecan.ca:/scratch/ssilber/results ./
```
