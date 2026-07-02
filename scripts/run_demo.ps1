param(
  [int]$Channels = 64,
  [int]$Height = 56,
  [int]$Width = 56,
  [int]$Kernel = 3,
  [int]$Stride = 1,
  [int]$Pad = 1,
  [int]$Filters = 64,
  [int]$BlockRows = 64,
  [int]$Threads = 8,
  [int]$Warmup = 2,
  [int]$Iters = 10,
  [string]$Backend = 'both',
  [switch]$NoVerify
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$buildDir = Join-Path $rootDir 'build'

cmake -S $rootDir -B $buildDir
cmake --build $buildDir --config Release

$exePath = Join-Path $buildDir 'Release/darknet_xitao_dgemm_demo.exe'
if (-not (Test-Path $exePath)) {
  $exePath = Join-Path $buildDir 'darknet_xitao_dgemm_demo.exe'
}

$args = @(
  '--channels', $Channels,
  '--height', $Height,
  '--width', $Width,
  '--kernel', $Kernel,
  '--stride', $Stride,
  '--pad', $Pad,
  '--filters', $Filters,
  '--block-rows', $BlockRows,
  '--threads', $Threads,
  '--warmup', $Warmup,
  '--iters', $Iters,
  '--backend', $Backend
)

if ($NoVerify) {
  $args += '--no-verify'
}

& $exePath $args
