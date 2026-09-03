基本公式
接受率：
* $\text{accept\_ratio}_{\text{HEAD}} = \frac{N_{\text{accepted}}}{N_{\text{draft}}}
$
* 示例：假设第一个 HEAD 推理 100 个 step，其中 90个 step 被接收，则接受率=90/100 = 90%

平均接收长度：
* $\text{avg\_accept\_len} = 1 + \text{accept\_ratio}_{\text{HEAD1}} + \text{accept\_ratio}_{\text{HEDA1}} \cdot \text{accept\_ratio}_{\text{HEAD2}}+ \cdots$
* 示例：假设MTP有三个头，每个头的接受率均为80%，则平均接收长度=1 + 0.8 + 0.8*0.8 + 0.8*0.8*0.8 =2.952

推理耗时：
* w/o MTP：只包含主模型推理的耗时
    * $T_{target}$：Target Model 推理一个step耗时

* w/   MTP：同时包含主模型验证与 Draft Model 推理耗时  $T_{target\_verify}$与$T_{draft}$
    *  $T_{target\_verify}$ ：主模型推理（验证）一个step的耗时，相比原始 $T_{target}$一般会有1.0~1.5倍耗时膨胀
    * $T_{draft}$：Draft Model 推理耗时(MTP)


加速比计算
单位时间内产出 Token 数量
* w/o MTP:$Throughput_{baseline} = \frac{T_{total}}{T_{target}} = \frac{T_{total}}{T_{target}} * 1$
* w/ MTP:  $Throughput_{mtp} = \frac{T_{total} * avg\_accept\_len}{T_{target\_verify} + T_{draft}} = \frac{T_{total}}{T_{target\_verify} + T_{draft}} * avg\_accept\_len$

加速比（单位时间内产出Token数量对比）:
* $speed\_up = \frac{Throughput_{mtp}}{Throughput_{baseline}} = \frac{T_{target} * avg\_accept\_len}{{(T_{target\_verify} + T_{draft}) * 1}}$
* 示例：以目前 EB5正式版 PD分离(TP4DP16EP64)为例
    * 三步接受率分别为 80% 74% 67%
    * 接受长度计算



单步接受率
累计接收长度
Baseline
/
1
单步
80%
1 + 0.8 = 1.8
两步
74%
(1 + 0.8 + 0.8*0.74) = 2.39
三步
67%
(1 + 0.8 + 0.8*0.74 + 0.8 * 0.74 * 0.67) = 2.79
    * 加速比为


Target 单步耗时(ms)
Draft 耗时(ms)
总耗时(ms)
总接受长度
理论加速比(VS BaseLine)
理论加速比(VS上一步)
Baseline
76
/
76
 /
1.0
1.0
单步
90
8
98
(1+0.8)  =  1.8
(1.8/1) / (98/76) = 1.39
(1.8/1) / (98/76) = 1.39
两步
104
14
118
(1+0.8+0.59) = 2.4
(2.4/1)  / (118/76) = 1.55
(2.4/1.8)  / (118/98) = 1.1
三步
117
22
139
(1+0.8_0.59+0.4) = 2.8
(2.8/1)  / (139/76) =1.53
(2.8/2.4)  / (139/118) =0.99