| # | 模型 | 开源 | 许可证 | 权重可得性 | 落盘状态 |
|---|---|---|---|---|---|
| 1 | RPCA | yes | - | no_weights | no_weights |
| 2 | RPCA+DRNN | no | - | reimplement | reimplement |
| 3 | Conv-TasNet | yes | MIT / 无许可证（见 repos） | available | partial(0/3) |
| 4 | DPRNN (DPRNN-TasNet) | yes | Apache-2.0 / 无许可证（见 repos） | custom_weights | custom_weights |
| 5 | MMDenseLSTM | yes | 无许可证（仓库无 LICENSE 文件） | available | downloaded |
| 6 | BS-RoFormer L12 | yes | MIT | available | downloaded |
| 7 | BS-RoFormer L6 | yes | MIT | available | partial(1/2) |
| 8 | Band-Split RNN (BSRNN) | yes | MIT | available | partial(0/3) |
| 9 | IRM / IBM oracle | yes | MIT | no_weights | no_weights |
| 10 | MDX-Net | yes | MIT | available | partial(1/6) |
| 11 | Pac-HuBERT-SEP | no | - | unavailable | unavailable |

### 权重文件明细

| 模型 | 权重 | 期望大小 | 实际大小 | 大小 | md5 |
|---|---|---|---|---|---|
| Conv-TasNet | Conv-TasNet MUSDB18 4sec_L20 | - | - | - | - |
| Conv-TasNet | Conv-TasNet MUSDB18 8sec_L20 | - | - | - | - |
| Conv-TasNet | Conv-TasNet MUSDB18 8sec_L64 | - | - | - | - |
| MMDenseLSTM | MMDenseLSTM MUSDB18 (paper) | 5.5MB | 5.5MB | OK | - |
| BS-RoFormer L12 | BS-RoFormer ep_317 (SDR 12.9755) | 609.7MB | 609.7MB | OK | - |
| BS-RoFormer L12 | ep_317 配置 | 2.0KB | 2.0KB | OK | - |
| BS-RoFormer L6 | BS-RoFormer ep_937 (SDR 10.5309) | 374.9MB | 270.2MB | MISMATCH | - |
| BS-RoFormer L6 | ep_937 配置 | 2.4KB | 2.4KB | OK | - |
| Band-Split RNN (BSRNN) | bsrnn-opt | 1.7GB | 64.3MB | MISMATCH | MISMATCH |
| Band-Split RNN (BSRNN) | bsrnn-large | 1.5GB | 60.4MB | MISMATCH | MISMATCH |
| Band-Split RNN (BSRNN) | simo-bsrnn-opt | 1.1GB | 51.3MB | MISMATCH | MISMATCH |
| MDX-Net | KUIELab-MDX-Net Track A (onnx_A.zip) | 105.2MB | 36.9MB | MISMATCH | MISMATCH |
| MDX-Net | mixer.ckpt | 1.2KB | 1.2KB | OK | OK |
| MDX-Net | mdx_extra/e51eebcc | - | - | - | - |
| MDX-Net | mdx_extra/a1d90b5c | - | - | - | - |
| MDX-Net | mdx_extra/5d2d6c55 | - | - | - | - |
| MDX-Net | mdx_extra/cfa93e08 | - | - | - | - |
