$src = 'E:\AgentNate'
$dst = 'E:\test_agentnate'

$excludeDirs = @(
    'python', 'node', 'node_modules', 'modules', 'envs',
    'vllm-source', '.n8n-instances', '__pycache__', '.claude',
    'temp', '_archive', '.vscode', '.idea', '.git'
)

$excludeFiles = @(
    'settings.json', '.mcp.json', 'package-lock.json',
    '*.log', '*.pid', '*.pyc', '*.pyo'
)

function Copy-Filtered {
    param($Source, $Destination, $ExcludeDirs, $ExcludeFiles)

    if (!(Test-Path $Destination)) {
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    }

    # Copy files in current directory
    Get-ChildItem -Path $Source -File | ForEach-Object {
        $skip = $false
        foreach ($pattern in $ExcludeFiles) {
            if ($_.Name -like $pattern) { $skip = $true; break }
        }
        if (!$skip) {
            Copy-Item $_.FullName -Destination $Destination
        }
    }

    # Recurse into subdirectories
    Get-ChildItem -Path $Source -Directory | ForEach-Object {
        if ($ExcludeDirs -notcontains $_.Name) {
            Copy-Filtered -Source $_.FullName -Destination (Join-Path $Destination $_.Name) -ExcludeDirs $ExcludeDirs -ExcludeFiles $ExcludeFiles
        }
    }
}

Copy-Filtered -Source $src -Destination $dst -ExcludeDirs $excludeDirs -ExcludeFiles $excludeFiles
Write-Output "Copy complete."

# Verify
$fileCount = (Get-ChildItem -Path $dst -Recurse -File | Measure-Object).Count
$dirCount = (Get-ChildItem -Path $dst -Recurse -Directory | Measure-Object).Count
Write-Output "Result: $fileCount files in $dirCount directories"

# Check excluded dirs don't exist
foreach ($d in @('python','node','node_modules','modules','envs')) {
    $p = Join-Path $dst $d
    if (Test-Path $p) {
        Write-Output "WARNING: $d was NOT excluded!"
    } else {
        Write-Output "OK: $d excluded"
    }
}
