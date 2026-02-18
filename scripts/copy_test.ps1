$src = "E:\AgentNate"
$dst = "E:\test_agentnate"

$excludeDirs = @(
    "python", "node", "node_modules", "modules", "envs",
    "vllm-source", ".n8n-instances", "__pycache__", ".claude",
    "temp", "_archive", ".vscode", ".idea", ".git"
)

$excludeFiles = @(
    "settings.json", ".mcp.json", "package-lock.json",
    "*.log", "*.pid", "*.pyc", "*.pyo"
)

# Build robocopy args
$args = @($src, $dst, "/E", "/NP")

foreach ($d in $excludeDirs) {
    $args += "/XD"
    $args += (Join-Path $src $d)
}

foreach ($f in $excludeFiles) {
    $args += "/XF"
    $args += $f
}

Write-Host "Copying source files..."
Write-Host "robocopy $($args -join ' ')"
& robocopy @args

$exitCode = $LASTEXITCODE
if ($exitCode -lt 8) {
    Write-Host "Copy complete (robocopy exit: $exitCode)"
} else {
    Write-Host "Copy FAILED (robocopy exit: $exitCode)"
    exit 1
}
