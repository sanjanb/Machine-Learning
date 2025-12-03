Param(
    [string]$WorkspacePath = "D:\Projects\Teaching\Machine-Learning"
)

Write-Host "Starting ipynb -> markdown watcher for: $WorkspacePath"

# Resolve Python executable (prefer local venv)
$venvPython = Join-Path $WorkspacePath ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }

function Convert-Notebook([string]$filePath) {
    try {
        if (!(Test-Path $filePath)) { return }
        $dir = Split-Path $filePath -Parent
        $name = Split-Path $filePath -Leaf
        $outputName = "notebook"  # always write notebook.md in same folder
        Write-Host "Converting: $filePath -> $dir\$outputName.md"
        & $pythonExe -m jupyter nbconvert --to markdown $filePath --output $outputName | Out-String | Write-Host
    }
    catch {
        Write-Warning "Conversion failed for ${filePath}: $($_.Exception.Message)"
    }
}

# Initial bulk conversion (optional): convert all existing .ipynb files
Get-ChildItem -Path $WorkspacePath -Filter "*.ipynb" -Recurse | ForEach-Object { Convert-Notebook $_.FullName }

# Set up a filesystem watcher
$fsw = New-Object System.IO.FileSystemWatcher
$fsw.Path = $WorkspacePath
$fsw.Filter = "*.ipynb"
$fsw.IncludeSubdirectories = $true
$fsw.EnableRaisingEvents = $true

# Debounce to avoid duplicate rapid events
$pending = New-Object 'System.Collections.Concurrent.ConcurrentDictionary[string,DateTime]'
$thresholdMs = 500

function Should-ProcessEvent([string]$path) {
    $now = Get-Date
    if ($pending.ContainsKey($path)) {
        $last = $pending[$path]
        if ((New-TimeSpan -Start $last -End $now).TotalMilliseconds -lt $thresholdMs) { return $false }
        $null = $pending.TryUpdate($path, $now, $last)
        return $true
    } else {
        $null = $pending.TryAdd($path, $now)
        return $true
    }
}

Register-ObjectEvent -InputObject $fsw -EventName Changed -Action {
    $path = $Event.SourceEventArgs.FullPath
    if (Should-ProcessEvent $path) { Convert-Notebook $path }
} | Out-Null

Register-ObjectEvent -InputObject $fsw -EventName Created -Action {
    $path = $Event.SourceEventArgs.FullPath
    if (Should-ProcessEvent $path) { Convert-Notebook $path }
} | Out-Null

Register-ObjectEvent -InputObject $fsw -EventName Renamed -Action {
    $path = $Event.SourceEventArgs.FullPath
    if (Should-ProcessEvent $path) { Convert-Notebook $path }
} | Out-Null

Write-Host "Watching for .ipynb changes under $WorkspacePath (Press Ctrl+C to stop)"

# Keep the script running
while ($true) { Start-Sleep -Seconds 1 }
