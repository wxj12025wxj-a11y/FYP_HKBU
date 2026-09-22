# fetch_bsrnn.ps1  (v2 -- segmented download with explicit Range + .NET append)
#
# Why not `curl -C -`:
#   On Zenodo, if the server ignores Range and returns 200 (full body), curl APPENDS
#   the whole file to the existing content -> the file grows past the expected size.
#   The old oversize branch used Remove-Item, which fails while curl still holds the
#   handle -> infinite loop. Also: two concurrent writers on the same file caused chaos.
#
# v2 strategy (one curl per round, write to a separate .part file, then append):
#   1. read current length `cur` of the main file, download with `--range cur-` to <name>.part
#   2. if part length == expected size -> server ignored Range; replace main file directly
#   3. else append part to main file (.NET FileStream), delete part
#   4. if main file exceeds expected size -> delete and restart (no curl holds it then)
#   5. when size matches, verify MD5; on mismatch delete and restart
#
# NOTE: all messages are ASCII on purpose. PowerShell 5.1 reads BOM-less .ps1 as ANSI,
#       so non-ASCII string literals get mangled and break parsing.
#
# Usage: powershell -ExecutionPolicy Bypass -File E:\FYP_HKBU\tools\fetch_bsrnn.ps1 -Only bsrnn-opt.zip

param(
    [int]$MaxAttempts = 600,
    [int]$StallLimit  = 30,
    [string]$Only = ""
)

$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'

$W  = 'E:\FYP_HKBU\weights\BSRNN'
$LG = 'E:\FYP_HKBU\logs'
New-Item -ItemType Directory -Force -Path $W | Out-Null
New-Item -ItemType Directory -Force -Path $LG | Out-Null

$items = @(
    [pscustomobject]@{ Name='bsrnn-opt.zip';      Url='https://zenodo.org/api/records/17516442/files/bsrnn-opt.zip/content';      Size=1827060285; Md5='89075aac776f82295a074a56d0428a18' },
    [pscustomobject]@{ Name='bsrnn-large.zip';    Url='https://zenodo.org/api/records/17516442/files/bsrnn-large.zip/content';    Size=1627580581; Md5='0d1e298e725bf45577a4f61e0a30c2b1' },
    [pscustomobject]@{ Name='simo-bsrnn-opt.zip'; Url='https://zenodo.org/api/records/17516442/files/simo-bsrnn-opt.zip/content'; Size=1213413202; Md5='817e8ce4bcd17c38bd3812167e97db58' }
)

if ($Only -ne "") { $items = @($items | Where-Object { $_.Name -eq $Only }) }
if ($items.Count -eq 0) { Write-Output "no manifest entry matches -Only $Only"; exit 1 }

$tag = 'all'
if ($Only -ne "") { $tag = $Only }
$log = Join-Path $LG ("_fetch_bsrnn_" + $tag + ".txt")

function Write-Log([string]$msg) {
    Add-Content -Path $log -Encoding utf8 -Value ("[" + (Get-Date -Format 'HH:mm:ss') + "] " + $msg)
}

function Get-Len([string]$p) {
    if (Test-Path $p) { return (Get-Item $p).Length }
    return 0
}

function Append-File([string]$dst, [string]$part) {
    $out = [System.IO.File]::Open($dst, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write)
    try {
        $in = [System.IO.File]::OpenRead($part)
        try { $in.CopyTo($out) } finally { $in.Close() }
    } finally { $out.Close() }
}

function Md5-Of([string]$p) {
    return (Get-FileHash -Algorithm MD5 -Path $p).Hash.ToLower()
}

Write-Log ("=== fetch_bsrnn v2 start attempts=" + $MaxAttempts + " stallLimit=" + $StallLimit + " ===")

foreach ($it in $items) {

    $dst  = Join-Path $W $it.Name
    $part = $dst + ".part"
    $done = $false
    $stall = 0

    for ($a = 1; $a -le $MaxAttempts; $a++) {

        $cur = Get-Len $dst

        if ($cur -eq $it.Size) {
            $h = Md5-Of $dst
            if ($h -eq $it.Md5) { $done = $true; break }
            Write-Log ($it.Name + " size ok but md5 mismatch (" + $h + ") -> redownload")
            Remove-Item $dst -Force -ErrorAction SilentlyContinue
            $cur = 0
        }

        if ($cur -gt $it.Size) {
            Write-Log ($it.Name + " oversize " + $cur + " > " + $it.Size + " -> delete and restart")
            Remove-Item $dst -Force -ErrorAction SilentlyContinue
            $cur = 0
        }

        # part is a CROSS-ROUND accumulator: never delete it here, or a mid-round
        # failure would throw away everything downloaded in that round.
        $off = $cur + (Get-Len $part)
        $chunk = $part + ".new"
        if (Test-Path $chunk) { Remove-Item $chunk -Force -ErrorAction SilentlyContinue }

        & curl.exe -L --fail --connect-timeout 30 --range ("{0}-" -f $off) -o $chunk $it.Url 2>$null | Out-Null

        $clen = Get-Len $chunk

        # even on a failed/aborted round, keep whatever bytes did arrive
        if ($clen -gt 0) {
            if ($clen -eq $it.Size) {
                # server ignored Range and sent the whole file
                if (Test-Path $dst) { Remove-Item $dst -Force -ErrorAction SilentlyContinue }
                Remove-Item $part -Force -ErrorAction SilentlyContinue
                Move-Item -Force $chunk $dst
                Write-Log ($it.Name + " server returned whole file -> replaced")
            }
            else {
                Append-File $part $chunk
                Remove-Item $chunk -Force -ErrorAction SilentlyContinue
            }
        }
        else {
            Remove-Item $chunk -Force -ErrorAction SilentlyContinue
        }

        # flush the part into the main file once it is fully assembled
        $total = $cur + (Get-Len $part)
        if ($total -eq $it.Size) {
            if ($cur -eq 0) {
                Move-Item -Force $part $dst
            }
            else {
                Append-File $dst $part
                Remove-Item $part -Force -ErrorAction SilentlyContinue
            }
        }

        $new = Get-Len $dst + (Get-Len $part)
        if ($new -le $cur) {
            $stall = $stall + 1
            Write-Log ($it.Name + " round " + $a + ": no net growth (" + $cur + " -> " + $new + ")")
            if ($stall -ge $StallLimit) { Write-Log ($it.Name + " giving up"); break }
            Start-Sleep -Seconds 3
        }
        elseif (($a % 3) -eq 1) {
            $pct = [math]::Round(100.0 * $new / $it.Size, 1)
            Write-Log ($it.Name + " " + $pct + "% (" + $new + "/" + $it.Size + ")")
        }
    }

    if ($done) {
        Write-Log ($it.Name + " DONE md5=" + (Md5-Of $dst))
    }
    else {
        Write-Log ($it.Name + " INCOMPLETE at " + (Get-Len $dst) + "/" + $it.Size)
    }
}

Write-Log "=== fetch_bsrnn v2 done ==="
Get-Content $log -Tail 25
