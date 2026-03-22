# Cluster Quick Commands (Slurm)

Use these from:

```bash
cd /home/PERSONALE/juanpablo.bayonapen2/InhomogeneousQuench/C_pp
```

## 1) Check job status

```bash
# All your jobs
squeue -u $USER

# One specific job
squeue -j <JOBID>

# Pending jobs + reason
squeue -u $USER -t PD -o "%.10i %.12P %.20j %.2t %.40R"

# Predicted start (if available)
squeue --start -j <JOBID>

# Full job details
scontrol show job <JOBID>
```

## 2) Check partitions and node availability (before submitting)

```bash
# Partition summary
sinfo -s

# Partition states
sinfo -o "%.12P %.5a %.10l %.6D %.10t"

# Node-level view with free CPUs and features
sinfo -N -o "%.20N %.12P %.10t %C %f"

# Matrix partitions only
sinfo -N -p m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11 -o "%N %P %t %C %f"

# B partitions only
sinfo -N -p b1,b2,b3,b4,b5,b6 -o "%N %P %t %C %f"
```

Interpret `%C` as `alloc/idle/other/total`.

## 3) Submit jobs

```bash
sbatch <script.batch>

# Example: choose partitions + resources
sbatch -p b2,b3,b4 --cpus-per-task=16 --time=24:00:00 <script.batch>
```

## 4) Watch logs (.out/.err)

```bash
# Live tail specific job
tail -f logs/ghd_it_heat_<JOBID>.out
tail -f logs/ghd_it_heat_<JOBID>.err

# Last 80 lines
tail -n 80 logs/ghd_it_heat_<JOBID>.out
tail -n 80 logs/ghd_it_heat_<JOBID>.err
```

### Check output progress while running

```bash
# Latest generated CSVs (change OUTDIR as needed)
ls -lt runs/<OUTDIR_NAME>/GHD_IT_CSV | head

# How many CSVs have been produced so far
find runs/<OUTDIR_NAME>/GHD_IT_CSV -maxdepth 1 -type f -name "*.csv" | wc -l
```

### Tail latest job automatically

```bash
jid=$(squeue -u "$USER" -h -o "%i" | head -n1)
tail -f logs/ghd_it_heat_${jid}.out
```

```bash
jid=$(squeue -u "$USER" -h -o "%i" | head -n1)
tail -f logs/ghd_it_heat_${jid}.err
```

## 5) After completion

```bash
# Accounting summary
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed

# If job failed: inspect logs and reason
scontrol show job <JOBID> | egrep "JobState=|Reason=|Partition=|TimeLimit=|NumCPUs=|Constraint="
```

## 6) Cancel jobs

```bash
# Cancel one
scancel <JOBID>

# Cancel all your pending jobs
scancel -u $USER -t PD
```
