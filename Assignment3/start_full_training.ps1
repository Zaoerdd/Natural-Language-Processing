$ErrorActionPreference = "Stop"

$assignmentDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $assignmentDir
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$workDir = Join-Path $assignmentDir "work"
$tokenizerPath = Join-Path $workDir "wikizh_tokenizer_bytelevel_52000.json"
$rawTextDir = Join-Path $workDir "raw_txt"
$tokenShardDir = Join-Path $workDir "token_shards_full"
$trainOutputDir = Join-Path $workDir "train_full"
$compareJson = Join-Path $workDir "compare_formal.json"
$tokenCountJson = Join-Path $workDir "token_count_summary.json"
$sampleText = [string]::Concat(
    [char]0x592A,
    [char]0x9633,
    [char]0x7167,
    [char]0x5E38,
    [char]0x5347,
    [char]0x8D77,
    [char]0x3002
)

if (-not (Test-Path $python)) {
    throw "Python executable not found: $python"
}

if (-not (Test-Path $tokenizerPath)) {
    throw "Tokenizer file not found: $tokenizerPath"
}

if (-not (Test-Path $rawTextDir)) {
    throw "Raw text directory not found: $rawTextDir"
}

New-Item -ItemType Directory -Force -Path $workDir | Out-Null
New-Item -ItemType Directory -Force -Path $trainOutputDir | Out-Null

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "==> Generating tokenizer comparison output"
& $python (Join-Path $assignmentDir "compare_tokenizers.py") `
    --tokenizer $tokenizerPath `
    --text $sampleText `
    --output_json $compareJson

if (-not (Test-Path (Join-Path $tokenShardDir "manifest.json"))) {
    Write-Host "==> Preparing full token shards"
    & $python (Join-Path $assignmentDir "prepare_tokenized_shards.py") `
        --input $rawTextDir `
        --tokenizer $tokenizerPath `
        --output_dir $tokenShardDir
}
else {
    Write-Host "==> Token shards already exist, skipping shard preparation"
}

Write-Host "==> Counting tokens from full token shards"
& $python (Join-Path $assignmentDir "count_tokens.py") `
    --input $tokenShardDir `
    --mode tokens `
    --output_json $tokenCountJson

$resumeCheckpoint = $null
$interruptedCkpt = Join-Path $trainOutputDir "checkpoint_interrupted.pt"
if (Test-Path $interruptedCkpt) {
    $resumeCheckpoint = $interruptedCkpt
}
else {
    $latestStep = Get-ChildItem -Path $trainOutputDir -Filter "checkpoint_step_*.pt" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($latestStep) {
        $resumeCheckpoint = $latestStep.FullName
    }
}

$trainArgs = @(
    (Join-Path $assignmentDir "run_pretrain.py")
    "--data_file", $tokenShardDir
    "--tokenizer", $tokenizerPath
    "--output_dir", $trainOutputDir
    "--n_epochs", "1"
    "--batch_size", "3"
    "--vocab_size", "52000"
    "--eval_freq", "500"
    "--save_ckpt_freq", "5000"
    "--log_freq", "20"
    "--warmup_steps", "2000"
    "--num_workers", "0"
    "--seed", "123"
)

if ($resumeCheckpoint) {
    Write-Host "==> Resuming training from $resumeCheckpoint"
    $trainArgs += @("--resume_from", $resumeCheckpoint)
}
else {
    Write-Host "==> Starting fresh 1-epoch training run"
}

Write-Host "==> Launching training"
& $python @trainArgs

Write-Host "==> Pipeline finished"
