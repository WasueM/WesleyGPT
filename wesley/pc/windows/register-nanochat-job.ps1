# Wesley wrote this
# "nanochat-job" Scheduled Task: runs ~/jobs/runjob.sh inside WSL, hidden, as wesle,
# at logon (so training resumes after a reboot) and on demand. Remove with:
#   Unregister-ScheduledTask -TaskName nanochat-job -Confirm:$false
$action  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -Command wsl.exe -d Ubuntu -u wesley -e bash /home/wesley/jobs/runjob.sh"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "wesle"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId "wesle" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "nanochat-job" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Select TaskName, State | Format-Table -Auto
