"""Build the requested detailed Markdown report from immutable experiment records."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'docs/实验路线与结果总报告_2026-11-02.md'
R1=ROOT/'experiments/gpu_stage_optimization'
R2=ROOT/'experiments/temporal_robust_round2/results'
parts=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def add(text):parts.append(text.strip()+'\n\n')
def num(v):return f'{v:.8f}'
def link(p,label=None):
    p=Path(p);assert p.exists(),p
    relative=os.path.relpath(p,OUT.parent).replace('\\','/')
    return f'[{label or p.name}]({relative})'
def table(headers,rows):
    def cell(v):return str(v).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']
    lines+=['| '+' | '.join(cell(x) for x in row)+' |' for row in rows]
    add('\n'.join(lines))
def result1(name,folder='results',seed=42):return read(R1/folder/name/f'seed{seed}/result.json')
def result2(name,seed=42,scope='development'):return read(R2/name/f'seed{seed}'/scope/'result.json')
def why(checks):
    labels={'mae_0p1_percent':'MAE改善不足0.1%','mean_mae_0p1_percent':'跨范围平均MAE改善不足0.1%',
            'mse_no_worse':'MSE升高','scope_mse_no_worse':'至少一个范围MSE升高',
            'both_periods':'开发早晚时段未同时达标','both_dev_periods':'开发早晚时段未同时达标',
            'all_block_mae':'至少一个日期块MAE退化超限','content_increment':'内容增量检查未通过',
            'content':'内容增量检查未通过'}
    failures=[]
    for k,v in checks.items():
        if v:continue
        failures.append(labels.get(k,'未超过匹配对照 '+k.removeprefix('beats_') if k.startswith('beats_') else k))
    return '；'.join(failures) or '全部登记门槛通过'
def new_config_table(names,folder='results'):
    rows=[]
    for name,change in names:
        r=result1(name,folder);s=r['development']
        rows.append([f'`{name}`',change,num(s['mae']),num(s['mse']),r['best_epoch'],r['epochs_run']])
    table(['配置','本次改动或对照角色','开发MAE','开发MSE','最佳epoch','实际epoch'],rows)
def decision_table(decisions):
    table(['阶段','匹配基线 → 候选','采用结果','判定原因'],[
        [d['stage'],f"`{d['baseline']}` → `{d['candidate']}`",f"保留 `{d['selected']}`",why(d['checks'])]
        for d in decisions])
def scope_matrix(names):
    rows=[]
    for name in names:
        records=[result2(name,scope=s) for s in ('rolling_A','rolling_B','development')]
        values=[num(r['validation']['mae'])+' / '+num(r['validation']['mse']) for r in records]
        epochs=' / '.join(str(r['best_epoch']) for r in records)
        rows.append([f'`{name}`',*values,epochs])
    table(['配置','rolling_A MAE / MSE','rolling_B MAE / MSE','开发MAE / MSE','最佳epoch A / B / 开发'],rows)

def main():
    scores=read(HERE/'scores.json');models={r['id']:r for r in scores['models']}
    runs=read(HERE/'runs.json');assert len(runs)==148
    market=read(R2/'market_selection.json');modal=read(R2/'modal_selection.json');fixed=read(R2/'FINAL_SELECTION.json')
    freeze=read(R2/'TEST_FREEZE.json');b2=read(R1/'ensemble/final_summary.json')
    add('# 实验路线与逐轮结果报告')
    add('更新日期：2026-11-02。范围：既有基础研究、两批新增GPU实验，以及冻结候选的内部留出评估。')
    add('本报告按**实验迭代轮次**组织。每轮说明“为什么改、具体改什么、与谁比较、结果如何、是否采用”。训练中的单个epoch不是一次独立实验；新增148次训练的最佳epoch和实际epoch全部列在附录，逐epoch曲线仍保存在对应的 `history.json`。本次仅整理既有记录，没有新增训练或重新运行测试推理。')
    add('## 1 最终成绩先明确')
    table(['成绩口径','模型','开发MAE','开发MSE','内部留出MAE','内部留出MSE'],[
        [label,models[k]['name'],num(models[k]['development']['mae']),num(models[k]['development']['mse']),num(models[k]['internal_holdout']['mae']),num(models[k]['internal_holdout']['mse'])]
        for k,label in [('E6','历史最佳研究结果'),('GPU1_B2','第一批GPU最终固定组合'),('GPU2_market','最后一轮冻结行情模型'),('GPU2_multimodal','最后一轮多模态实验配置')]])
    add('**目前最好已知内部留出研究成绩：E6平稳化DLinear，MAE 0.35034519、MSE 0.32020074。最后一轮实际冻结模型是GRU，MAE 0.36252380、MSE 0.33565344，未刷新历史成绩。**')
    add('这里的“最好”只是在已列完整模型固定平均中的历史观测结果，不能当成新盲测选模。E6是纯行情模型；最新多模态配置最终新闻/财报输出头均为0，实际等同于行情GRU。两者均不能直接写成已通过有效三模态要求的新正式方案。**官方隐藏测试成绩与课程最终总分尚未产生**，也不自行把MAE/MSE换算成百分制。')
    add('## 2 任务、数据和轮次总览')
    table(['项目','固定口径'],[
        ['输入与目标','过去96个交易日的行情、新闻、财报 → 未来24日标准化Close'],
        ['训练 / 开发 / 隔离 / 内部留出','587,587 / 42,930 / 18,630 / 15,390条'],
        ['内部留出日期','目标起始日2024-01-24至2024-02-27；目标最晚2024-03-29'],
        ['边界','开发目标最晚2024-01-23；与留出目标日期不重叠'],
        ['指标','样本与24个预测位置等权的MAE、MSE，越低越好'],
        ['平均方式','先平均固定种子的预测，再算误差；如为“种子分数均值”会另外注明'],
        ['数据限制','内部留出已在历史实验中揭晓，不是官方隐藏测试或新的独立盲测']])
    table(['阶段','模型/输入路线','研究问题'],[
        ['E0','最后值基线','不学习时序也能得到的参照'],['E1','DLinear','线性残差是否有效'],
        ['E2','Close单变量GRU','非线性记忆是否有效'],['E3','七变量GRU','其他行情字段是否提供增量'],
        ['E4','PatchTST','局部时间片注意力是否有效'],['E5','iTransformer','变量间注意力是否有效'],
        ['E6','原始/平稳化特征对照，历史骨干为多变量DLinear','输入表示的影响'],
        ['E7','行情加新闻','语义是否超过篇数与覆盖信息'],['E8','行情加财报','数值是否超过掩码与年龄'],
        ['E9','行情加新闻加财报','联合是否超过匹配单模态'],['E10','轻量频域融合','频谱结构是否超过简单新闻分支']])
    add('PatchTST和iTransformer都属于Transformer。E6是实验编号，不是新的Transformer架构。不同目录中同名的 `E6_stationary` 使用不同训练协议和权重，不能当成同一个模型。')
    table(['新增GPU轮次','配置与训练次数','目的'],[
        ['G1.1 初始阶段优化','23个seed42配置，23次训练','E1–E10匹配对照'],
        ['G1.2 针对性后续','9个seed42配置，9次训练','修复alpha规则、收缩、近期新闻和财务比率'],
        ['G1.3 固定组合复验','补seed43/44的4个组件，8次训练','检验B2固定等权组合'],
        ['G2.1 行情时间验证','12个配置 × 3个范围 = 36次','局部尺度、容量与跨时段稳定性'],
        ['G2.2 反射补充对照','6个配置 × 3个范围 = 18次','消除无条件预测漂移'],
        ['G2.3 新闻/财报/频域','8个配置 × 3个范围 = 24次','受控骨干上的模态增量'],
        ['G2.4 联合训练','2个配置 × 3个范围 = 6次','模态dropout策略'],
        ['G2.5 三种子复验','4个组件 × 2个新增种子 × 3范围 = 24次','最后两个固定候选复验'],
        ['新增合计','40 + 108 = 148次','不把既有历史CPU实验重复计入']])
    add('历史协议与两批GPU协议的优化器、损失、预算、校准和平均方式并非完全一致；跨批次成绩提供性能参照，单项改进是否有效以同轮匹配对照判断。')

    add('## 3 历史研究轮次')
    add('### H1 原版DLinear与窗口初筛')
    add('**问题与改动：** 建立Close-only DLinear，比较移动平均窗口25、15、5，固定seed42和学习率1e-3，再对窗口25补seed43/44；另将学习率降至1e-4排查优化幅度。')
    table(['配置','开发MAE','开发MSE','结果'],[
        ['E0最后值','0.217307','0.131206','固定基线'],
        ['窗口25，seed42','0.236111','0.146795','三个窗口中最好，但明显差于E0'],
        ['窗口15，seed42','0.236219','0.146875','未改善'],['窗口5，seed42','0.236489','0.147121','未改善'],
        ['窗口25，三个种子分数均值','0.235596','0.146163','MAE/MSE较E0恶化约8.42%/11.40%'],
        ['窗口25，lr=1e-4，seed42','0.237021','0.147070','降低学习率未解决问题']])
    add('**决定：** 不把原版E1作为已验证升级。训练误差下降而开发误差升高，转向输入中心化和正则化。来源：'+link(ROOT/'experiments/e1_dlinear/RESULTS.md')+'。')

    add('### H2 DLinear中心化与L2正则化')
    add('**改动：** 输入先减历史最后一个Close，再做趋势/剩余项分解；比较Adam耦合L2为0、1e-3、1e-2。固定窗口25，其余同条件。')
    opt=read(ROOT/'experiments/e1_dlinear/results/optimization/summary.json')
    table(['seed42配置','开发MAE','开发MSE','最佳epoch'],[[r['variant'],num(r['development']['mae']),num(r['development']['mse']),r['best_epoch']] for r in opt['screening']])
    table(['选定L2=0.01的种子','开发MAE','开发MSE','最佳epoch'],[[r['seed'],num(r['development']['mae']),num(r['development']['mse']),r['best_epoch']] for r in opt['confirmation']])
    add('**结果与决定：** 三个种子的分数均值为MAE 0.217069、MSE 0.131303。相对原版大幅改善，但相对E0仅MAE略好，MSE略差，且seed44退化；保留中心化方向，继续检查残差校准，未宣称稳定超过基线。来源：'+link(ROOT/'experiments/e1_dlinear/OPTIMIZATION.md')+'。')

    add('### H3 稳定求解与训练日期近期加权')
    add('**改动：** 在中心化线性模型上，预设均匀训练权重、日期半衰期252和126，各配四个L2系数，共12个解析岭回归；再对均匀/252近期加权代表配置做全量平滑MAE+0.2MSE优化。损失和求解方式发生变化，因此不把本轮当成仅改变学习率的对照。')
    refinement=read(ROOT/'experiments/e1_dlinear/results/refinement/summary.json')
    table(['配置','求解/目标','开发MAE','开发MSE'],[[r['candidate'],r['solver'],num(r['development']['mae']),num(r['development']['mse'])] for r in refinement['all_candidates']])
    add('**结果与决定：** 14个代表配置都未同时超过E0；更充分地优化训练目标没有带来开发收益，不采用这批权重。来源：'+link(ROOT/'experiments/e1_dlinear/REFINEMENT.md')+'，完整数据见 '+link(ROOT/'experiments/e1_dlinear/results/refinement/summary.json')+'。')

    add('### H4 按预测期限收缩残差与权重平均')
    add('**问题与改动：** 短期残差过度修正，而第24日相对更好。对H2选中的固定权重比较四条预定曲线：不收缩、全部乘0.5、乘h/24、乘(h/24)²；没有自由搜索24个验证期系数。')
    cal=read(ROOT/'experiments/e1_dlinear/results/calibration/summary.json')
    table(['seed42残差曲线','开发MAE','开发MSE'],[[r['curve'],num(r['development']['mae']),num(r['development']['mse'])] for r in cal['screening_seed42']])
    table(['固定h/24后的种子','开发MAE','开发MSE'],[[r['seed'],num(r['development']['mae']),num(r['development']['mse'])] for r in cal['confirmation']])
    add('**决定：** 按seed42开发MAE选择h/24，补齐三种子且不挑成员；兼容线性模型等权平均权重，等价于平均预测，推理仍为一个4,656参数模型。开发MAE/MSE为 **0.21605229 / 0.13034210**。冻结后的内部留出为 **0.35129397 / 0.32407166**，相对E0改善约0.84%/1.90%。收缩已折入权重，不能重复乘。')
    add('来源：'+link(ROOT/'experiments/e1_dlinear/REFINEMENT.md')+'、'+link(ROOT/'experiments/e1_dlinear/HOLDOUT.md')+'。')

    add('### H5 七变量输入、平稳化表示与新闻分支')
    add('**改动：** E6把Close-only扩展为七变量DLinear，比较原始字段和平稳化特征；E7冻结同种子选定E6再训练新闻分支。')
    hist=read(ROOT/'experiments/e1_dlinear/e6_plus/results/summary.json')
    aggregates=[('E6原始七变量',hist['e6']['raw']['aggregate']),('E6平稳化',hist['e6']['stationary']['aggregate']),('E7平稳化加新闻',hist['e7']['aggregate'])]
    table(['配置','三个种子的MAE均值','三个种子的MSE均值'],[[n,num(a['mae']['mean']),num(a['mse']['mean'])] for n,a in aggregates])
    table(['固定预测平均','开发MAE','开发MSE','内部留出MAE','内部留出MSE'],[[models[k]['name'],num(models[k]['development']['mae']),num(models[k]['development']['mse']),num(models[k]['internal_holdout']['mae']),num(models[k]['internal_holdout']['mse'])] for k in ('E6','E7')])
    add('**结果与决定：** E6按原规则选平稳化；E7在开发集改善匹配E6，但留出明显变差。新闻语义置零的变化仅约浮点量级，不能认定有语义增益。历史E6获得目前已知最低内部留出误差，但开发集仍弱于增强E1。这里“各种子指标均值”和“平均预测后指标”分别列出，不能混用。')
    add('来源：'+link(ROOT/'experiments/e1_dlinear/e6_plus/RESULTS.md')+'、'+link(ROOT/'experiments/e1_dlinear/e6_plus/HOLDOUT.md')+'。')

    add('### H6 PatchTST初版与固定收缩')
    add('**初版改动：** 用PatchTST捕获局部时间片，加入同输入、同训练条件DLinear控制；先只做seed42。PatchTST开发MAE/MSE为0.21715390/0.13504648，内部留出0.35947047/0.33998372，未形成升级。')
    add('**后续改动：** 保持原训练及选轮方式，补seed43/44，对固定三种子平均分别用残差系数1、0.5、0.25；系数不按种子改变。')
    table(['固定三种子预测平均','开发MAE','开发MSE','判定'],[
        ['DLinear控制 ×1','0.21716468','0.13211337','原控制'],['DLinear控制 ×0.5','0.21634668','0.13053666','作为后续匹配线性参照'],
        ['DLinear控制 ×0.25','0.21662021','0.13059066','未采用'],['PatchTST ×1','0.21692806','0.13283559','原预测'],
        ['PatchTST ×0.5','0.21586688','0.13045219','相对增强E1的MAE改善仅0.0858%，未达0.1%门槛'],
        ['PatchTST ×0.25','0.21629180','0.13043702','未达门槛']])
    add('**决定：** 收缩改善了原PatchTST，但没有通过相对较强参照的全部门槛，所以这轮没有重新评估内部留出。来源：'+link(ROOT/'experiments/e4_patchtst/RESULTS.md')+'、'+link(ROOT/'experiments/e4_patchtst_shrinkage/RESULTS.md')+'。')

    add('### H7 Transformer有限优化：选轮、宽度、dropout与变量注意力')
    add('**改动与匹配关系：** P0让固定×0.5的推理误差参与早停；P1仅将宽度128改64；P2在较好父配置上将dropout从0.1改0.3；I0改用七变量iTransformer并设同输入线性对照。P0同时采用8轮/patience3的新预算，与历史12轮/patience4不同，不能把全部差异归因于选轮规则。')
    screen=read(ROOT/'experiments/transformer_optimization_common/results/screen.json')
    table(['seed42配置','开发MAE','开发MSE','最佳epoch'],[[r['spec']['id'],num(r['development']['mae']),num(r['development']['mse']),r['best_epoch']] for r in screen['rows']])
    add('**复验与决定：** 按预定规则只给P2_dropout补seed43/44，同时补匹配线性模型。Transformer固定平均开发为 **0.21500354 / 0.12978113**，三个种子都胜过匹配线性MAE，通过开发门槛。冻结后留出为 **0.35368771 / 0.32801956**；匹配线性为0.35288808/0.32643317。没有超过历史E6，不能称为留出升级。')
    add('来源：'+link(ROOT/'experiments/transformer_optimization_common/RESULTS.md')+'。')

    add('### H8 历史财报、联合与频域扩展')
    add('**改动：** 在选定行情骨干上依次加入财报残差、新闻与财报联合分支、轻量SpecTF风格频域融合。')
    table(['固定三种子预测平均','开发MAE','开发MSE','内部留出MAE','内部留出MSE'],[[models[k]['name'],num(models[k]['development']['mae']),num(models[k]['development']['mse']),num(models[k]['internal_holdout']['mae']),num(models[k]['internal_holdout']['mse'])] for k in ('E8','E9','E10')])
    add('**决定：** 都未在该留出上超过E6。完整内容置零的变化很小，历史财报E8为0.35037913/0.32023650，虽接近E6，也不能写成超过。某次消融更低不是可事后选择的新完整模型。')

    add('### H9 模态编码器正则化修复')
    add('**问题与改动：** 旧模态编码器参数发生明显收缩。F1/F2、M1/M2、S1/S2分别比较AdamW全部参数衰减与仅矩阵衰减。均为seed42、20轮上限，与旧40轮预算不同；先修复优化现象，不假定修复后必然提分。')
    table(['步骤','分支/改动','选中开发MAE','选中开发MSE','最佳epoch','实际训练轮最低MAE（不含epoch0）','决定'],[
        ['F1','财报，全部衰减','0.21882429','0.13244591',0,'0.22173370','未通过MAE与内容门槛'],
        ['F2','财报，仅矩阵衰减','0.21882429','0.13244591',0,'0.22173118','未通过MAE与内容门槛'],
        ['M1','联合，全部衰减','0.21882429','0.13244591',0,'0.22266335','未通过MAE、内容与时段门槛'],
        ['M2','联合，仅矩阵衰减','0.21882429','0.13244591',0,'0.22260639','未通过MAE、内容与时段门槛'],
        ['S1','频域，全部衰减','0.21882429','0.13244591',0,'0.22515725','未通过MAE、内容与时段门槛'],
        ['S2','频域，仅矩阵衰减','0.21882429','0.13244591',0,'0.22515261','未通过MAE、内容与时段门槛']])
    add('F步骤的强参照为seed42 E6（0.21882429/0.13244591）；M/S步骤的较强参照为seed42 E7（0.21764661/0.13185552）。修复后末轮编码器不再塌缩，但开发最优都回到纯行情E6的epoch0，因此没有可采用的预测增益。未给这六步追加43/44、未新增留出测分；各自原计划后续F3/M3/S3未在该轮执行。')
    add('来源：'+ '、'.join(link(ROOT/f'experiments/{folder}/RESULTS.md') for folder in ('e8_dlinear_reports_refine','e9_dlinear_multimodal_refine','e10_dlinear_spectf_refine'))+'。')

    add('## 4 第一批GPU G1.1：23个初始配置')
    add('**共同协议：** 全587,587条训练样本，seed42，最多12轮、patience3、batch512，AdamW lr=0.0003，矩阵wd=0.01，通常MAE+0.2MSE。相对匹配对照要求开发MAE至少降0.1%、MSE不升、早晚MAE不恶化超过0.5%；模态还需重训覆盖/掩码对照和内容诊断。')
    add('E7–E10在这批固定使用本批E6_stationary为受控行情骨干，没有把阶段E3的局部通过结果自动迁移进去；这批不能称为“最优GRU加新闻”的实验。')
    numerical=[('E1_baseline','中心化Close DLinear，MSE系数0.2'),('E1_mse05','仅将MSE系数改为0.5'),('E2_last','Close-only GRU64，最后隐藏状态'),('E2_mean','最后状态改平均池化'),('E3_baseline','原始七变量GRU，无通道dropout'),('E3_channel_drop','仅非Close通道dropout=0.1'),('E4_linear_control','PatchTST的匹配线性控制'),('E4_ff256','PatchTST128，两层，FF256，alpha0.5'),('E4_ff128','仅FF256缩为128'),('E5_layers2','iTransformer两层'),('E5_layers1','仅减为一层'),('E6_raw','原始七变量DLinear'),('E6_stationary','改平稳化七变量')]
    modal1=[('E7_baseline','冻结E6，简单新闻分支'),('E7_l2_normalized','512维日新闻逐向量L2归一化'),('E7_coverage','重训仅数量/掩码的匹配控制'),('E8_mlp','54维财报MLP门控残差'),('E8_linear','财报编码和输出改线性'),('E8_coverage','重训仅财报掩码/年龄控制'),('E9_joint','从零联合训练新闻和财报分支'),('E9_pretrained','单模态预训练，冻结编码器，仅训门控与输出'),('E10_modes13','轻量频域，13个模式'),('E10_modes7','频率模式13减7')]
    add('### 4.1 全部数值配置结果');new_config_table(numerical)
    add('### 4.2 全部模态配置与控制结果');new_config_table(modal1)
    add('### 4.3 各阶段是否采用')
    decisions=[read(R1/f'results/decisions/E{i}_seed42.json') for i in range(1,11)]
    decision_table(decisions)
    add('**本轮实际有效的局部改动：** E3通道dropout相对匹配无dropout版本，MAE改善0.5720%、MSE也下降，通过seed42阶段门槛；但未进入本批最终三种子复验，不能写成最终稳定提升。')
    add('**不能被低总分掩盖的负结果：** E9_pretrained开发MAE 0.21325422低于后来的B2，但早晚时段、匹配单模态和内容门槛未通过，因此未直接采用。它的epoch0已经含有单模态预训练权重，不等于“完全没有训练”；其他零初始化模型的epoch0通常为最后值回退。')
    add('E7归一化有内容响应，但相对基线MAE只改善0.0148%，且未完整胜过覆盖控制。E8线性头相对MLP改善0.2354%，但仍未超过财报结构控制、内容检查失败，因此也未采用。')
    add('来源：'+link(R1/'PLAN.json')+'；表格每个配置的逐epoch记录见其 `history.json`，逐次结果链接见附录。')

    add('## 5 第一批GPU G1.2：9个针对性后续配置')
    add('**为什么继续：** E1/E6常回退epoch0；发现E4首轮把alpha0.5同时用于训练和推理，与历史有效规则“训练1、推理0.5”不同；新闻内容与覆盖控制接近，财报值缺少增量。因此在训练前另登记9个有限配置，没有无限扩网格。')
    follow=[('T1_E1_half','E1残差固定乘0.5，训练推理一致'),('T2_E6_raw_half','原始七变量DLinear残差乘0.5'),('T3_E6_stationary_half','平稳化七变量DLinear残差乘0.5'),('T4_Patch_reference','修复：训练alpha1，早停/推理alpha0.5，FF256'),('T5_Patch_ff128','在T4规则下仅FF256减128'),('T6_News_recent20','仅保留最近20交易日新闻与覆盖信息'),('T7_News_recent20_coverage','重训T6的语义置零覆盖控制'),('T8_Report_ratios','同表财务比率与log总资产，仍固定180日可用性'),('T9_Report_ratio_masks','重训T8的数值置零掩码/年龄控制')]
    new_config_table(follow,'followup_results')
    follow_decisions=[read(p) for p in sorted((R1/'followup_results/decisions').glob('*.json'))]
    decision_table(follow_decisions)
    add('**逐项结论：** T1/T2/T3均回退epoch0，收缩没有解决本批线性模型的泛化问题。T4修复后恢复训练收益；T5相对T4的MAE只下降0.0361%，且MSE略升，不能采用容量改动。T6没有优于原完整窗口新闻，也未优于匹配近期覆盖控制。T8虽然优于较弱的旧MLP，却差于T9掩码控制，所以财报数值增量仍不成立。')
    add('E4容量结论以修复后的T4/T5匹配实验为准；最初两个epoch0的E4结果不能单独被解释为PatchTST架构无效。来源：'+link(ROOT/'experiments/gpu_targeted_followup/PLAN.json')+'。')

    add('## 6 第一批GPU G1.3：固定组合筛选与三种子复验')
    add('**改动：** 不再增加复杂融合网络，只预登记两个等权预测组合；不搜索连续权重。B1=完整新闻+新闻覆盖；B2=B1再加财报掩码/年龄分支。')
    screen1=read(R1/'ensemble/screen.json')
    table(['seed42组合','开发MAE','开发MSE','登记门槛'],[[k,num(v['development']['mae']),num(v['development']['mse']),'通过' if v['passed'] else '未通过'] for k,v in screen1['candidates'].items()])
    add('两个都通过初筛。B2的MAE更低，MSE高于B1但仍低于登记的完整新闻参照，因此按事先规定的开发MAE主规则选B2，不是按后来的留出分数选组合。')
    table(['新增复验组件','种子','开发MAE','开发MSE','最佳epoch','实际epoch'],[[r['id'],r['seed'],num(r['development']['mae']),num(r['development']['mse']),r['best_epoch'],r['epochs_run']] for r in [read(p) for p in sorted((R1/'replications').glob('*/seed*/result.json'))]])
    table(['种子/固定平均','B2开发MAE','B2开发MSE','匹配完整新闻MAE','匹配完整新闻MSE'],[
        [seed,num(v['candidate']['mae']),num(v['candidate']['mse']),num(v['matched_news']['mae']),num(v['matched_news']['mse'])] for seed,v in b2['single_seed_scores'].items()]+[
        ['三个种子平均预测',num(b2['development']['mae']),num(b2['development']['mse']),num(b2['matched_news_average']['mae']),num(b2['matched_news_average']['mse'])]])
    add('**结果与解释：** 三个种子均改善匹配完整新闻分支，最终B2共9个小模型、596,529参数。关闭新闻语义后开发MAE为0.21506083，关闭财报结构后为0.21464989，表明最终组合依赖这些输入；这不等于财报数值有效，也不能替代重训内容控制。')
    add('本批三个E6父模型均选真正零残差epoch0，因此行情主干实际是最后Close锚点；新增开发收益不能归因于DLinear学到了新的历史动态。财报有效输入限于可用性、缺失掩码和报告年龄。')
    add('### 6.1 冻结B2后按追加要求测试')
    table(['同一固定B2','MAE','MSE'],[['开发集',num(models['GPU1_B2']['development']['mae']),num(models['GPU1_B2']['development']['mse'])],['内部留出',num(models['GPU1_B2']['internal_holdout']['mae']),num(models['GPU1_B2']['internal_holdout']['mse'])]])
    add('**决定：** 留出相对E0的MAE/MSE恶化约2.45%/2.67%；开发集收益没有延续到该时段。原融合计划本身不查看留出；在完成冻结后，根据后续明确的测试要求追加评估，单独保存记录，未倒改原计划或据此改权重。')
    add('来源：'+link(ROOT/'experiments/gpu_targeted_followup/ENSEMBLE_PLAN.json')+'、'+link(R1/'ensemble/final_summary.json')+'、'+link(R1/'holdout_20261101/summary.json')+'。')

    add('## 7 第二批GPU：先改时间验证，再做优化')
    add('**为什么重开一批：** 第一批B2开发更好而留出失败。新批次把跨时段稳定性放在最低单个开发总分之前，优化期间不再测内部留出。')
    table(['范围','训练样本','验证样本','时间边界'],[['rolling_A','440,221','29,152','训练完整目标结束早于2023-01-03；验证起始日2023-01-03至2023-02-28'],['rolling_B','535,747','33,210','训练完整目标结束早于2023-07-03；验证起始日2023-07-03至2023-08-31'],['development','587,587','42,930','原完整训练集与开发集，隔离边界不变']])
    add('**共同改动：** 价格按输入窗口最后值居中，并除以窗口Close标准差；量能用窗口内log-volume中心化；行情残差限制为0.5倍窗口尺度，每个辅助分支为0.1倍。默认窗口归一化Huber损失，E1另设MAE对照；AdamW lr=0.0002、wd=0.05，batch512，最多8轮、patience2。')
    add('**选轮与采用：** 每个范围按目标起始日分三块，以“最差块MAE/E0 + 0.25×最差块MSE/E0”选轮，并保留epoch0。匹配候选跨三范围的平均MAE至少降0.1%，每范围MSE不升，每块MAE退化不超过0.25%；全局行情资格另受E0块MAE/MSE上限约束。日期块仍可能标签重叠，只是压力检查，不是三个新盲测。')

    add('### G2.1 行情优化：12配置 × 3范围 = 36次')
    table(['阶段','基线','候选改动'],[['E1','E1_MAE','E1_Huber：MAE改Huber'],['E2','E2_GRU64','E2_GRU16：隐藏层64缩16'],['E3','E3_drop01','E3_drop03：非Close通道dropout 0.1升0.3'],['E4','E4_full','E4_small：d_model128/层2/FF256整体缩为64/1/128'],['E5','E5_layers2','E5_layers1：iTransformer两层减一层'],['E6','E6_raw','E6_stationary：原始窗口表示改平稳化窗口表示']])
    names=['E1_MAE','E1_Huber','E2_GRU64','E2_GRU16','E3_drop01','E3_drop03','E4_full','E4_small','E5_layers2','E5_layers1','E6_raw','E6_stationary']
    scope_matrix(names)
    decision_table(market['decisions'][:6])
    add('**决定：** 只有E4容量策略通过本阶段跨范围匹配门槛，但两版在完整开发范围均选择epoch0，不能说其最终全量模型学到了新收益。按全局行情资格与最差范围目标排序，最终骨干选 `E3_drop01`：七变量GRU16、通道dropout0.1。隐藏层64改16这一对比本身未通过，不把“最终选用GRU16”误写成缩宽度因果收益已证实。')

    add('### G2.2 价格反射约束：6配置 × 3范围 = 18次')
    add('**观察与改动：** 早期E1/E6、部分GRU和PatchTST仍回退epoch0，有界幅度本身未消除无条件漂移。另在训练前登记六个补充配置，使用相同骨干同时计算正向与价格反射输入，抵消无条件截距/漂移；与各自同结构非反射配置比较，不假定真实市场完全对称。')
    reflections=['S_E1_Huber','S_E2_GRU16','S_E3_drop03','S_E4_small','S_E5_layers1','S_E6_stationary']
    scope_matrix(reflections)
    decision_table(market['decisions'][6:])
    add('**决定：** 六个都未通过完整匹配门槛。S_E4_small在部分时间范围有收益，但某范围MSE或日期块退化超限；不能只报告其平均MAE改善。完整开发范围六个均回退epoch0，未用于最终行情骨干。')

    add('### G2.3 新闻、财报与频域：8配置 × 3范围 = 24次')
    add('**改动与控制：** 固定每个范围已经选好的匹配GRU父模型。E7把每日展开输出头改为三个固定32日池化段，并重训覆盖-only控制；E8比较数值、同表比率及掩码/年龄控制；E10把交叉频谱模式13改3，并与简单E7比较。')
    modnames=['E7_daily','E7_pooled','E7_coverage','E8_values','E8_ratios','E8_masks','E10_modes13','E10_modes3']
    scope_matrix(modnames)
    decision_table(modal['decisions'])
    add('**决定：** 全部未通过完整模态验收。E7池化和E10降频的部分范围MSE/日期块可以守住，但总MAE改进或内容检查不足；E8比率未胜过匹配掩码控制。作为后续联合训练输入保留 `E7_daily` 和 `E8_values`，只是继续做受控实验，不代表它们已获内容增量认证。完整开发范围各配置都保留父模型输出。')
    add('**财报审计限制：** 旧数值缓存先做z-score并截断到±10，再反归一化。普通财报值在两个历史滚动范围未检出截断；比率缓存在rolling_A训练/验证分别有511/72个字段截断，rolling_B训练有729个，验证0个。截断不可逆，相关比率滚动实验仅作近似诊断，不能声称完全不含原训练期校准影响。')

    add('### G2.4 联合训练：2配置 × 3范围 = 6次')
    add('**改动：** 从已选新闻、财报独立分支开始联合训练，比较模态dropout 0.1与0.5；行情父模型冻结。要求超过匹配单模态并保留内容证据。')
    scope_matrix(['E9_dropout01','E9_dropout05'])
    decision_table(fixed['stage_E9'])
    add('**决定：** 保留预定基线 `E9_dropout01` 作为多模态实验候选；更强dropout没有通过平均MAE和内容门槛。`FINAL_SELECTION.json` 中 `multimodal_accepted=false`，它在测试前就没有通过完整研究验收，不能在测试后改称合格。')

    add('### G2.5 固定组件三种子复验：24次')
    add('**改动：** 成员和结构不再改动，补seed43/44的行情E3_drop01、新闻E7_daily、财报E8_values、联合E9_dropout01，每个组件分别训练三个范围，共24次。最终两个候选都按42/43/44固定等权平均，不挑最好种子。')
    table(['组件','种子','开发MAE','开发MSE','最佳epoch','实际epoch'],[[name,seed,num(r['validation']['mae']),num(r['validation']['mse']),r['best_epoch'],r['epochs_run']] for seed in (43,44) for name in ('E3_drop01','E7_daily','E8_values','E9_dropout01') for r in [result2(name,seed)]])
    add('上表列完整开发范围；两组历史范围的全部16次新增运行也列在附录。三种子平均预测的验证结果如下。')
    table(['固定候选','范围','MAE','MSE'],[[label,scope,num(v['mae']),num(v['mse'])] for label,c in freeze['candidates'].items() for scope,v in c['validation_scores'].items()])
    add('**复验结果：** 三个最终全量种子的新闻与财报输出头均为0；行情与多模态候选在完整开发集、随后留出集的预测完全相同。历史rolling_B的辅助分支曾有局部收益，不代表最终全量权重实际使用了它们。')

    add('## 8 全部优化后统一测试的最终结果')
    add('只有上述所有阶段和三种子复验完成，独立导出、无标签样本核对与模型哈希冻结通过后，才按预登记清单对两个固定候选统一测试。每个固定候选只测一次，之后没有训练、改权重或事后改选成员。')
    table(['内部留出模型','MAE','MSE'],[[models[k]['name'],num(models[k]['internal_holdout']['mae']),num(models[k]['internal_holdout']['mse'])] for k in ('E0','E1','E6','PatchTST','GPU1_B2','GPU2_market','GPU2_multimodal')])
    add('**最终判定：** 最后一轮GRU为0.36252380/0.33565344，相对E0恶化约2.33%/1.61%；没有达到提分目标。它比上一个失败B2的部分指标略低，也不能据此写成超过基线。历史最佳研究结果仍为E6的0.35034519/0.32020074，但这是已知内部留出成绩，且E6是纯行情模型。')
    add('### 8.1 分期限结果')
    last=read(R2/'final_holdout/summary.json')['results']['market']['score']
    table(['预测位置','最终GRU MAE','最终GRU MSE','E0 MAE','E0 MSE'],[[h,num(last['horizons'][h]['mae']),num(last['horizons'][h]['mse']),num(last['e0']['horizons'][h]['mae']),num(last['e0']['horizons'][h]['mse'])] for h in ('1','6','12','24')])
    add('### 8.2 技术校验与模型状态')
    table(['检查项','实际结果'],[['独立导出','model.py与model.pt，无参数build_model严格重载'],['原始无标签样本','每候选48个开发样本去除y后核对，最大预测差约2.38e-7'],['CPU/GPU与缺失模态','对齐通过；辅助模态全缺失仍输出有限[B,24]'],['评分实现','NumPy与独立PyTorch float64归约一致，差小于1e-12'],['权重身份','冻结前后哈希相同；148份回传最佳检查点逐一核对'],['真实三模态依赖','最新三种子新闻/财报输出头全0，未通过有效依赖'],['正式交付资格','技术可运行；尚无兼具已验证提分与有效三模态依赖的新方案']])

    add('## 9 每轮结论如何累计')
    table(['轮次','确实改进了什么','没有解决什么','后续决定'],[
        ['H1→H2','中心化大幅降低原DLinear开发误差','未稳定同时超过E0两指标','继续残差校准'],
        ['H3','更充分求解了训练目标','求解更好不等于开发更好','不采用解析/稳健分支权重'],
        ['H4','期限收缩与固定平均使增强E1小幅超过E0','只在已用开发和一段留出成立','保留历史可复现基线'],
        ['H5','平稳化E6取得目前最低已知留出误差','新闻语义收益未证明，开发与留出排名反转','把内容与覆盖拆开验证'],
        ['H6→H7','校准感知选轮、dropout改善PatchTST开发结果','留出未超过E6','不宣称Transformer全面胜出'],
        ['H8→H9','修复编码器过度收缩','选中权重仍回退、没有内容增益','后续重训结构-only控制'],
        ['G1.1','E3通道dropout获单种子局部收益','多个阶段仍epoch0；模态未过内容门槛','登记9个针对性配置'],
        ['G1.2','修复E4训练/推理alpha不一致','近期新闻、财报比率仍未胜控制','只做有限等权组合'],
        ['G1.3','B2开发三种子均改善匹配新闻模型','开发收益没有延续到内部留出','重建历史时间切分验证'],
        ['G2.1→G2.2','更严格时间门槛，选择小型GRU','反射约束未通过完整门槛','固定骨干做模态比较'],
        ['G2.3→G2.5','完整控制、三范围与三种子复验','最终模态输出头归零','冻结后如实统一测试'],
        ['最终统一测试','明确记录失败而非只报开发最低分','仍未超过E0或历史E6','保留实验产物，不冒称正式升级']])
    add('后续若继续研究，应先固定未揭示时段与训练截止点、从原始财报生成不经旧截断的特征，再比较必要的小容量受控分支。当前已知内部留出不再是可用于反复搜索后声称独立泛化的证据。')

    add('## 10 结果文件与复现入口')
    add('以下链接以本Markdown所在 `docs/` 为起点；实验原始记录不因整理而覆盖。')
    source_files=[HERE/'scores.json',HERE/'runs.json',ROOT/'experiments/e1_dlinear/RESULTS.md',ROOT/'experiments/e1_dlinear/OPTIMIZATION.md',ROOT/'experiments/e1_dlinear/REFINEMENT.md',ROOT/'experiments/transformer_optimization_common/RESULTS.md',R1/'PLAN.json',ROOT/'experiments/gpu_targeted_followup/PLAN.json',ROOT/'experiments/gpu_targeted_followup/ENSEMBLE_PLAN.json',R1/'ensemble/final_summary.json',R1/'holdout_20261101/summary.json',ROOT/'experiments/temporal_robust_round2/PLAN.json',ROOT/'experiments/temporal_robust_round2/FOLLOWUP_PLAN.json',R2/'market_selection.json',R2/'modal_selection.json',R2/'FINAL_SELECTION.json',R2/'TEST_FREEZE.json',R2/'final_holdout/summary.json',R2/'MODEL_AUDIT.json',R2/'AUDIT_LIMITATIONS.json']
    add('\n'.join('- '+link(p,p.relative_to(ROOT).as_posix()) for p in source_files))
    add('本次整理已对E0、增强E1、历史E6/E7、B2及最新两套候选共7组保存预测与原始标签复算；MAE/MSE与既有记录差不超过1e-12，并核对历史E6、B2和最新候选检查点哈希。它是已有分数核对，不是新增模型测试或新选模。')

    add('## 附录 A 两批GPU全部148次运行')
    add('每一行是一份独立训练记录。配置名链接到该次 `result.json`；同目录 `protocol.json` 给出预算与配置，`history.json` 给出逐epoch曲线。`最佳epoch=0`表示采用初始化/预训练起点回退，不代表实际没有训练。第一批E9预训练起点与真正零残差初始化必须分别解释。')
    for group,caption in [('gpu1','A.1 第一批40次'),('gpu2','A.2 第二批108次')]:
        add('### '+caption)
        subset=[r for r in runs if r['round']==group]
        assert len(subset)==(40 if group=='gpu1' else 108)
        table(['编号','配置','seed','验证范围','MAE','MSE','最佳epoch','实际epoch'],[[i,link(ROOT/r['source'],r['configuration']),r['seed'],r['scope'],num(r['mae']),num(r['mse']),r['best_epoch'],r['epochs_run']] for i,r in enumerate(subset,1)])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(''.join(parts),encoding='utf-8')
    print(OUT)
    print('lines',len(OUT.read_text(encoding='utf-8').splitlines()),'runs',len(runs))

if __name__=='__main__':main()
