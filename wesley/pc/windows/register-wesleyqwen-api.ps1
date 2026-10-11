# Wesley wrote this
# "wesleyqwen-api" Scheduled Task: runs ~/wesleyqwen-api.sh (the home-PC model API) inside WSL, hidden, as wesle,
# at boot, at logon and on demand. S4U ("run whether the user is logged on or not") is what lets the boot trigger
# work after a restart that nobody logs in to, as the pipeline's MangumPcClaudeBridgeLoop task does. Its live wsl.exe client also keeps WSL from idling shut. It listens on 127.0.0.1:8090
# only; publishing it to the internet is a separate, deliberate step Wesley runs himself:
#   & "C:\Program Files\Tailscale\tailscale.exe" funnel --bg 8090     (undo: ... funnel --https=443 off)
# Remove with:  Unregister-ScheduledTask -TaskName wesleyqwen-api -Confirm:$false
$action  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -Command wsl.exe -d Ubuntu -u wesley -e bash /home/wesley/wesleyqwen-api.sh --port 8090"
$triggers = @((New-ScheduledTaskTrigger -AtStartup), (New-ScheduledTaskTrigger -AtLogOn -User "wesle"))
# Restart if it dies (a crash, or WSL being shut down), every minute, for a day.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -StartWhenAvailable -RestartCount 1440 -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId "wesle" -LogonType S4U -RunLevel Limited
Register-ScheduledTask -TaskName "wesleyqwen-api" -Action $action -Trigger $triggers -Settings $settings -Principal $principal -Force | Select TaskName, State | Format-Table -Auto
Start-ScheduledTask -TaskName "wesleyqwen-api"
