"""Unified route/results report; all scores come from existing audited predictions."""
from pathlib import Path
import json
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE=Path(__file__).resolve().parent
REPO=BASE/'Multimodal-Financial-Time-Series-Forecasting-main/Multimodal-Financial-Time-Series-Forecasting-main'
SUMMARY=REPO/'experiments/experiment_summary_20261102'
R1=REPO/'experiments/gpu_stage_optimization'
R2=REPO/'experiments/temporal_robust_round2/results'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
data=read(SUMMARY/'scores.json');models={m['id']:m for m in data['models']}
fixed=read(R2/'FINAL_SELECTION.json');freeze=read(R2/'TEST_FREEZE.json')
market=read(R2/'market_selection.json');modal=read(R2/'modal_selection.json')
style_source=(BASE/'report_build_20261101.py').read_text(encoding='utf-8').split("doc.add_paragraph('多模态金融预测进展与分阶段优化计划','Title')")[0]
ns={'__file__':str(BASE/'report_build_20261101.py')};exec(compile(style_source,str(BASE/'report_build_20261101.py'),'exec'),ns)
doc=ns['doc'];sec=ns['sec'];para=ns['para'];heading=ns['heading'];sub=ns['sub'];table=ns['table']
def f(value):return f'{value:.6f}'
def r1(name,folder='results'):return read(R1/folder/name/'seed42/result.json')
def r2(name,seed=42,scope='development'):return read(R2/name/f'seed{seed}'/scope/'result.json')

doc.core_properties.title='实验路线与结果总报告'
doc.core_properties.subject='历史研究与两批 GPU 优化的统一成绩核对'
doc.add_paragraph('实验路线与结果总报告','Title')
doc.add_paragraph('2026年11月2日  历史基线与两批GPU优化汇总','Subtitle')
para('截至目前，历史最佳内部留出研究成绩属于E6平稳化DLinear：MAE为0.35034519，MSE为0.32020074。最后一轮预先冻结的GRU模型成绩为0.36252380和0.33565344，两者属于不同实验记录，不能互换。',True)
para('本报告整合既有基础研究、第一批40次GPU训练和第二批108次GPU训练。新增训练共148次；此前历史CPU实验不重复计入。今天只汇总并复算已保存的预测，未新增训练或模型推理。')
table(['应使用的成绩名称','模型与平均方式','内部留出MAE','内部留出MSE'],[
 ['历史最佳研究成绩','E6 DLinear 三种子预测平均','0.35034519','0.32020074'],
 ['第一批GPU最终组合','B2 三成员乘三种子','0.36296970','0.33916921'],
 ['最后一轮冻结模型','小型GRU 三种子预测平均','0.36252380','0.33565344'],
 ['官方最终成绩','官方隐藏测试','尚未产生','尚未产生']],[1.65,2.45,1.5,1.5])
sub('本次确定的分数口径')
para('汇报“目前最好成绩”时，使用E6的0.35034519与0.32020074，并注明它是历史已知内部留出结果。汇报“最后一轮最终模型”时，使用GRU的0.36252380与0.33565344，明确该轮没有超过E0基线。')
para('课程正式成绩仍未知：目前没有官方隐藏测试结果，也没有可据此换算为百分制总分的已确认评分公式。MAE和MSE分别报告，不自行相加或转换成课程分数。')
para('新一轮多模态配置的新闻与财报输出头最终均为0，实际预测等同于行情GRU。它没有形成有效三模态依赖。历史最佳E6也是纯行情模型，不能直接等同于通过课程三模态要求的正式提交。')

heading('1 任务数据与整体实验路线',True)
para('任务使用过去96个交易日的行情、逐篇新闻向量和财报，预测未来24日标准化Close。主指标为MAE、MSE，24个预测位置等权。训练样本587,587条；开发样本42,930条；隔离边界18,630条；内部留出15,390条。')
table(['阶段','模型或输入','需要回答的问题'],[
 ['E0','最后值基线','完全不训练能达到什么水平'],
 ['E1','DLinear','线性时序残差能否提分'],
 ['E2','单变量GRU','非线性记忆是否优于线性'],
 ['E3','多变量GRU','其他行情字段是否提供增量'],
 ['E4','PatchTST','局部时间片与注意力是否有效'],
 ['E5','iTransformer','变量间注意力是否有效'],
 ['E6','行情特征表示对照','原始与平稳化输入如何影响预测'],
 ['E7','行情加新闻','语义是否超过篇数和覆盖信息'],
 ['E8','行情加财报','数值是否超过掩码与报告年龄'],
 ['E9','行情加新闻加财报','联合融合能否超过匹配单模态'],
 ['E10','轻量频域融合','频谱结构能否超过简单新闻分支']],[.55,2.35,4.2])
para('PatchTST和iTransformer均属于Transformer类模型。本轮实际比较了DLinear、GRU和这两类Transformer；最终统一测试的第二批候选使用GRU。E6本身是特征表示实验编号，历史实现采用多变量DLinear，并不是另一种Transformer。')
para('内部留出取公共val中最晚的预测起始日期，开发目标须全部早于留出开始。留出预测起始日为2024年1月24日至2月27日，共810只股票、19个起始日，目标最晚到3月29日。开发目标最晚到1月23日。')
para('该留出在历史研究中已被揭晓。其后结果用于如实记录跨时段表现，不能重新称为独立新盲测。官方隐藏测试文件没有提供。')

heading('2 既有研究与统一历史成绩',True)
para('以下使用同一标准化Close尺度和原始时间切分。除E0外，表中为预定三种子的固定预测平均；增强E1可等价合并线性权重。先平均预测再算误差，区别于先算各次分数再平均。')
table(['模型','开发MAE','开发MSE','留出MAE','留出MSE'],[
 [models[k]['name'],f(models[k]['development']['mae']),f(models[k]['development']['mse']),f(models[k]['internal_holdout']['mae']),f(models[k]['internal_holdout']['mse'])]
 for k in ('E0','E1','E6','E7','E8','E9','E10','PatchTST')],[2.0,1.275,1.275,1.275,1.275])
sub('历史路线的结论')
para('E1完成中心化、正则化、残差收缩和三种子复验。E6扩展为七变量并比较平稳化表示，在这段留出上取得表中最低的MAE与MSE；相对E0分别降低约1.11%和3.07%。这是描述性的历史最好结果，不据此事后挑选某个种子。')
para('优化PatchTST在开发集优于增强E1，但留出排序没有延续。E6开发集反而弱于增强E1和PatchTST。开发和留出排名反转，是后续加强时间验证的直接原因。')
para('历史新闻、财报、联合和频域模型都未在留出上超过E6。旧模态消融显示新闻语义或财报数值的响应很弱，因此不能把极小收益归因于内容有效。后续F1/F2、M1/M2、S1/S2正则化修复恢复了部分编码响应，但开发最佳检查点仍常回退epoch0，未形成可靠升级。')
para('历史E8为0.35037913与0.32023650，接近但未超过E6。不能因为某次“关闭新闻”消融或某个单种子更低，就将其改称最终新模型；这些结果不是独立重训的预定比较。')

heading('3 第一批GPU分阶段优化',True)
para('第一批包含23个初始配置、9个针对性后续配置和8次种子复验，共40次训练。使用完整官方训练集，主要按开发MAE下降至少0.1%、MSE不升、早晚开发段稳定及模态内容对照决定是否采用。训练在243完成，236完成独立推理验证。')
table(['阶段','验证内容与实际结果'],[
 ['E1','提高MSE权重并测试固定残差收缩；最优回到epoch0，未超过E0。'],
 ['E2','最后隐藏状态改平均池化；开发MAE由0.213691升至0.215064，保留最后状态。'],
 ['E3','非Close通道dropout为0.1；MAE由0.216061降至0.214825，MSE由0.136561降至0.134980。单种子局部通过。'],
 ['E4','修复历史训练与推理alpha规则后缩小前馈层；MAE 0.214783降至0.214705，改善仅0.036%，MSE略升，未采用。'],
 ['E5','iTransformer两层减一层；两版最优均回到epoch0。'],
 ['E6','原始与平稳化七变量、固定残差收缩；本批最优均回到epoch0。'],
 ['E7','新闻向量归一化及最近20日窗口；未稳定超过重训覆盖对照。'],
 ['E8','财报线性头和财务比率；比率MAE 0.214527，掩码对照0.214324，数值增量未通过。'],
 ['E9','单模态预训练再联合训练；开发MAE 0.213254，但时段和内容门槛未通过，未因最低总分直接采用。'],
 ['E10','频率模式13减7；最优回到epoch0，未超过简单新闻分支。']],[.55,6.55])
para('最后预先固定B2：完整新闻、新闻覆盖、财报结构三个成员，乘以42、43、44三个种子等权预测平均。开发MAE/MSE为0.21370155/0.12904843；实际内部留出为0.36296970/0.33916921，未泛化。')
para('B2的财报输入是掩码、可用性和报告年龄，未证明财报数值有效；其三个E6父模型均选零残差epoch0，行情主干实际是最后价格锚点。不能把这批开发收益归因于DLinear学到了新的历史动态。')

heading('4 第二批GPU时间稳定性优化',True)
para('第二批先训练22个阶段配置，再根据早期回退现象登记6个价格反射补充配置。28个配置分别训练两个历史时间切分与原开发范围，共84次；固定四个必要组件补seed43、44，各三个范围，共24次。合计108次训练、310个epoch。')
table(['验证范围','训练样本','验证样本','目标日期规则'],[
 ['rolling_A','440,221','29,152','训练目标结束早于2023年1月3日；验证起始日1月3日至2月28日。'],
 ['rolling_B','535,747','33,210','训练目标结束早于2023年7月3日；验证起始日7月3日至8月31日。'],
 ['development','587,587','42,930','全部官方训练样本与原开发切分。']],[1.15,1.05,1.05,3.85])
para('使用输入窗口局部尺度和有界残差，每个范围按最差日期块相对E0的MAE与MSE选轮；再以跨范围改进、每段退化上限决定采用。日期块可能有目标重叠，不能视为独立重复。六个反射补充模型用于消除无条件漂移，没有通过完整匹配门槛。')
table(['阶段','本批改动','跨时间匹配验收'],[
 ['E1','MAE与Huber损失','未通过'],['E2','GRU隐藏层64减16','未通过'],
 ['E3','七变量GRU16的dropout 0.1改0.3','未通过，保留0.1'],
 ['E4','PatchTST整体缩小容量','跨范围通过，开发仍为epoch0'],
 ['E5','iTransformer两层减一层','未通过'],['E6','窗口归一化原始与平稳化输入','未通过']],[.55,3.7,2.85])
para('综合时间验证选中E3_drop01：七变量GRU16、非Close通道dropout 0.1。随后E7比较每日展开与三段新闻池化，E8比较财报数值、比率和掩码，E9比较模态dropout 0.1/0.5，E10比较13/3个交叉频谱模式；均未通过完整模态验收。')
para('所有阶段和三种子复验结束后，才导出并冻结行情与多模态两套候选，再统一测试一次。最终两套预测完全一致，MAE为0.36252380、MSE为0.33565344，仍差于E0。测试后没有继续训练或调整成员。')

heading('5 最终模型分数与交付状态',True)
table(['固定实验对象','开发MAE','开发MSE','留出MAE','留出MSE'],[
 [models[k]['name'],f(models[k]['development']['mae']),f(models[k]['development']['mse']),f(models[k]['internal_holdout']['mae']),f(models[k]['internal_holdout']['mse'])]
 for k in ('E6','GPU1_B2','GPU2_market','GPU2_multimodal')],[2.0,1.275,1.275,1.275,1.275])
sub('最后一轮三种子复验')
table(['种子','GRU开发MAE','GRU开发MSE','多模态实际预测'],[
 [str(seed),f(r2('E3_drop01',seed)['validation']['mae']),f(r2('E3_drop01',seed)['validation']['mse']),'与对应行情模型相同'] for seed in (42,43,44)],[.6,1.6,1.6,3.3])
para('最后一轮开发成绩0.21459135/0.13041289是先平均三个种子的预测再评分，不是表中三个分数的算术平均。最终固定GRU在内部留出相对E0的MAE和MSE分别恶化约2.33%和1.61%。')
para('最终多模态权重审查发现，三个种子的新闻与财报输出头全部为0，因此任何有效输入下均退回行情预测。它没有通过有效三模态依赖；两个包只能作为实验产物归档，不能写成已验证提分的正式替代方案。',True)
sub('代码能够运行与研究有效性分别验收')
para('两套最新模型均完成独立model.py与model.pt导出、无参数build_model重载、48个去除y的原始开发样本核对、CPU/GPU一致性和缺失模态有限输出检查。原始样本预测最大差异约2.38e-7。技术接口通过，不代表模态内容有效或正式成绩合格。')
para('根目录model.py保留原模板，实验导出包分别存放。历史E6纯行情成绩、第一批B2输入结构信息的成绩，以及最新GRU成绩，都须保留各自模型身份。尚未得到兼具已验证提分和有效三模态依赖的新提交方案。')

heading('6 分数确认与复现证据',True)
para('本次直接读取原始val.npz标签、固定样本索引和已保存预测，独立以float64复算MAE及MSE。E0、增强E1、历史E6/E7、B2和最新两套候选共7组均与原记录一致，绝对差不超过1e-12；没有重新运行模型推理。')
para('同时核对历史E6三个检查点、B2导出文件以及最新两套导出文件的SHA256。汇总没有改动模型、预测、候选成员或选模结果。精确分数和校验哈希写入scores.json，148次运行明细写入runs.json。')
sub('证据文件')
entries=[
 ('experiments/experiment_summary_20261102/scores.json','统一成绩表、分数口径、保存预测复算及模型哈希。'),
 ('experiments/experiment_summary_20261102/runs.json','两批新增148次训练的配置、种子、范围、最佳轮数和分数。'),
 ('experiments/e1_dlinear/e6_plus/results/holdout/summary.json','历史E6及匹配E7、E1、E0的固定留出记录。'),
 ('experiments/gpu_stage_optimization/ensemble/final_summary.json','第一批B2的开发结果、模型组成和输入诊断。'),
 ('experiments/gpu_stage_optimization/holdout_20261101/summary.json','第一批B2的实际内部留出分数。'),
 ('experiments/temporal_robust_round2/results/FINAL_SELECTION.json','第二批测试前固定的候选与研究验收状态。'),
 ('experiments/temporal_robust_round2/results/TEST_FREEZE.json','最终候选哈希、等权方式和开发重载校验。'),
 ('experiments/temporal_robust_round2/results/final_holdout/summary.json','最后一轮统一测试的精确分数。'),
 ('experiments/temporal_robust_round2/results/MODEL_AUDIT.json','最新新闻与财报输出头归零的权重审计。')]
for path,note in entries:
 para(path,True,size=9.5).paragraph_format.space_after=ns['Pt'](1)
 para(note,size=10).paragraph_format.space_after=ns['Pt'](4)
para('财报缺少真实历史版本；旧比率缓存存在不可逆截断，滚动A验证涉及72个字段，相关比率滚动实验仅为近似诊断。已知内部留出已被反复观察，今后不能以此继续搜索后再声称独立泛化。',size=10)

footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=footer.add_run();field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');r._r.addnext(field)
for tree in (doc.element,doc.styles.element):
 for border in list(tree.xpath('.//w:pBdr')):border.getparent().remove(border)
OUT=REPO/'docs/实验路线与结果总报告_2026-11-02.docx'
doc.save(OUT);print(OUT)
