# download_all_weights.ps1
# 可续传的权重批量下载器（curl 8.x 并行模式）
# 用法: powershell -ExecutionPolicy Bypass -File E:\FYP_HKBU\tools\download_all_weights.ps1 [-Parallel 4] [-Only keyword]
param(
    [int]$Parallel = 4,
    [string]$Only = ""
)

$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'

$W  = 'E:\FYP_HKBU\weights'
$LG = 'E:\FYP_HKBU\logs'
New-Item -ItemType Directory -Force -Path $LG | Out-Null

# ---- 清单: Out(相对路径) / Url / Size(0=未知) / Md5(''=跳过) ----
$manifest = @(
    [pscustomobject]@{ Out='BS-RoFormer\model_bs_roformer_ep_317_sdr_12.9755.ckpt'; Url='https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt'; Size=639331213;  Md5='' },
    [pscustomobject]@{ Out='BS-RoFormer\model_bs_roformer_ep_937_sdr_10.5309.ckpt'; Url='https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt'; Size=393068365;  Md5='' },
    [pscustomobject]@{ Out='BSRNN\bsrnn-opt.zip';        Url='https://zenodo.org/api/records/17516442/files/bsrnn-opt.zip/content';        Size=1827060285; Md5='89075aac776f82295a074a56d0428a18' },
    [pscustomobject]@{ Out='BSRNN\bsrnn-large.zip';      Url='https://zenodo.org/api/records/17516442/files/bsrnn-large.zip/content';      Size=1627580581; Md5='0d1e298e725bf45577a4f61e0a30c2b1' },
    [pscustomobject]@{ Out='BSRNN\simo-bsrnn-opt.zip';   Url='https://zenodo.org/api/records/17516442/files/simo-bsrnn-opt.zip/content';   Size=1213413202; Md5='817e8ce4bcd17c38bd3812167e97db58' },
    [pscustomobject]@{ Out='MDX-Net\onnx_A.zip';         Url='https://zenodo.org/api/records/5717356/files/onnx_A.zip/content';            Size=110359032;  Md5='72605d3722ebef18c353f831abdbfc07' },
    [pscustomobject]@{ Out='MDX-Net\mixer.ckpt';         Url='https://zenodo.org/api/records/5717356/files/mixer.ckpt/content';            Size=1208;       Md5='485113a7e20a0dc5ccc47e7645464aed' },
    [pscustomobject]@{ Out='MDX-Net\mdx_extra\e51eebcc-c1b80bdd.th'; Url='https://dl.fbaipublicfiles.com/demucs/mdx_final/e51eebcc-c1b80bdd.th'; Size=0; Md5='' },
    [pscustomobject]@{ Out='MDX-Net\mdx_extra\a1d90b5c-ae9d2452.th'; Url='https://dl.fbaipublicfiles.com/demucs/mdx_final/a1d90b5c-ae9d2452.th'; Size=0; Md5='' },
    [pscustomobject]@{ Out='MDX-Net\mdx_extra\5d2d6c55-db83574e.th'; Url='https://dl.fbaipublicfiles.com/demucs/mdx_final/5d2d6c55-db83574e.th'; Size=0; Md5='' },
    [pscustomobject]@{ Out='MDX-Net\mdx_extra\cfa93e08-61801ae1.th'; Url='https://dl.fbaipublicfiles.com/demucs/mdx_final/cfa93e08-61801ae1.th'; Size=0; Md5='' }
)

if ($Only -ne "") { $manifest = @($manifest | Where-Object { $_.Out -like "*$Only*" }) }

# 跳过已完整下载的项（大小匹配），避免对完整文件重复发 Range 请求
$manifest = @($manifest | Where-Object {
    $dst = Join-Path $W $_.Out
    $skip = $false
    if ((Test-Path $dst) -and $_.Size -gt 0) {
        if ((Get-Item $dst).Length -eq $_.Size) { $skip = $true }
    }
    -not $skip
})

# 大文件优先，减少尾巴效应
$queue = @($manifest | Sort-Object -Property Size -Descending)

Write-Output ("[{0}] queue={1} parallel={2}" -f (Get-Date -Format 'HH:mm:ss'), $queue.Count, $Parallel)

$head = @('-Z',
          '--parallel-max', "$Parallel",
          '--parallel-immediate',
          '-L', '-C', '-',
          '--retry', '10', '--retry-delay', '5', '--retry-all-errors',
          '--connect-timeout', '30',
          '--create-dirs',
          '--no-progress-meter')

$tail = @()
foreach ($it in $queue) {
    $dst = Join-Path $W $it.Out
    $tail += @('-o', $dst, $it.Url)
}

& curl.exe @head @tail 2>&1 | Out-File -FilePath "$LG\_dlpar_out.txt" -Encoding utf8
$rc = $LASTEXITCODE
Add-Content -Path "$LG\_dl_all.log" -Value ("[{0}] PARALLEL rc={1}" -f (Get-Date -Format 'HH:mm:ss'), $rc)

# ---- 校验 ----
$lines = @()
foreach ($it in $queue) {
    $dst = Join-Path $W $it.Out
    if (-not (Test-Path $dst)) { $lines += ("MISSING  {0}" -f $it.Out); continue }
    $len = (Get-Item $dst).Length
    $ok = $true
    if ($it.Size -gt 0 -and $len -ne $it.Size) { $ok = $false }
    if ($it.Md5 -ne "") {
        $h = (Get-FileHash -Algorithm MD5 -Path $dst).Hash.ToLower()
        if ($h -ne $it.Md5) { $ok = $false }
    }
    if ($ok) { $tag = 'OK  ' } else { $tag = 'BAD ' }
    $lines += ("{0} {1,15} / {2,15}  {3}" -f $tag, $len, $it.Size, $it.Out)
}
$lines | Out-File -FilePath "$LG\_weights_verify.txt" -Encoding utf8
$lines | ForEach-Object { Write-Output $_ }
Write-Output "DONE"
