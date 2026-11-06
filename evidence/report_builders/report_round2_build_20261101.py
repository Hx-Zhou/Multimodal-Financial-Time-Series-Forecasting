"""Consolidated report after all round-two training and the frozen final test batch."""
from pathlib import Path
import json
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE=Path(__file__).resolve().parent
REPO=BASE/'Multimodal-Financial-Time-Series-Forecasting-main/Multimodal-Financial-Time-Series-Forecasting-main'
ROOT=REPO/'experiments/temporal_robust_round2/results'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
fixed=read(ROOT/'FINAL_SELECTION.json');freeze=read(ROOT/'TEST_FREEZE.json');test=read(ROOT/'final_holdout/summary.json')
market=read(ROOT/'market_selection.json');modal=read(ROOT/'modal_selection.json')
records=[read(p) for p in ROOT.glob('*/seed*/*/result.json')]
configs={r['id'] for r in records};epochs=sum(r['epochs_run'] for r in records)
style_source=(BASE/'report_build_20261101.py').read_text(encoding='utf-8').split("doc.add_paragraph('多模态金融预测进展与分阶段优化计划','Title')")[0]
ns={'__file__':str(BASE/'report_build_20261101.py')};exec(compile(style_source,str(BASE/'report_build_20261101.py'),'exec'),ns)
doc=ns['doc'];para=ns['para'];heading=ns['heading'];sub=ns['sub'];table=ns['table'];sec=ns['sec']
def record(name,scope='development',seed=42):return read(ROOT/name/f'seed{seed}'/scope/'result.json')
def fmt(v):return f'{v:.6f}'
def status(v):return '通过' if v else '未通过'

doc.core_properties.title='分阶段优化与最终测试复核'
doc.add_paragraph('分阶段优化与最终测试复核','Title')
doc.add_paragraph('2026年11月1日  第二轮完整优化与固定候选评估','Subtitle')
para(f'本轮完成E1至E10及针对性补充实验，共{len(configs)}个配置、{len(records)}次模型训练运行，累计{epochs}个训练epoch。全部优化、三种子复验及最终一次测试均已结束，但固定候选仍差于E0，本轮未实现可接受的测试集提分。',True)
mr=test['results']['market']['score'];mm=test['results']['multimodal']['score']
para(f"最终固定行情平均的留出MAE为{fmt(mr['mae'])}，MSE为{fmt(mr['mse'])}；固定多模态平均为{fmt(mm['mae'])}和{fmt(mm['mse'])}。下表同时保留基线及上一轮失败候选，所有指标均为标准化Close尺度，越低越好。")
table(['同一内部留出集','MAE','MSE'],[
 ['E0 最后值','0.354286','0.330340'],['历史增强E1','0.351294','0.324072'],
 ['历史E6三种子平均','0.350345','0.320201'],['历史优化PatchTST','0.353688','0.328020'],
 ['上一轮B2候选','0.362970','0.339169'],
 ['本轮固定行情平均',fmt(mr['mae']),fmt(mr['mse'])],
 ['本轮多模态实验配置',fmt(mm['mae']),fmt(mm['mse'])]],[4.1,1.5,1.5])
for label,score in [('行情',mr),('多模态',mm)]:
 d=[100*(score[k]/b-1) for k,b in [('mae',.35428627282119307),('mse',.33033977026568145)]]
 para(f'{label}候选相对E0的MAE变化为{d[0]:+.2f}%，MSE变化为{d[1]:+.2f}%；负数表示改善。')
para('本次测试共15,390条样本，预测起始日期为2024年1月24日至2月27日，目标最晚到3月29日。这是此前已揭晓的内部时间留出，官方隐藏测试集仍未公开；不能将本次结果描述为新的盲测。')

heading('1 验证方案和训练预算',True)
para('上轮B2在开发集有收益，却在内部留出上退化。本轮将跨时段稳定性放到较低的单一开发分数之前，优化期间只使用训练和开发范围，最后两个候选的成员、种子与等权规则均在测试前冻结。')
scope_rows=[]
for scope,note in [('rolling_A','训练目标结束早于2023年1月3日；验证起始日为1月3日至2月28日。'),('rolling_B','训练目标结束早于2023年7月3日；验证起始日为7月3日至8月31日。'),('development','使用全部官方训练样本；验证使用原42,930条开发样本，保留原隔离边界。')]:
 p=read(ROOT/fixed['market']/'seed42'/scope/'protocol.json')
 scope_rows.append([scope,str(p['training_samples']),str(p['validation_samples']),note])
table(['时间范围','训练样本','验证样本','边界规则'],scope_rows,[1.2,1.05,1.05,3.8])
sub('优化与选轮规则')
para('每配置seed42最多8轮、patience为2、batch为512。AdamW学习率0.0002、权重衰减0.05。每个时间范围内再按预测起始日划三段，以最差一段的MAE/E0加0.25倍最差MSE/E0选检查点，保留epoch0最后值参照。三段目标可能重叠，因此只是压力检查。')
para('跨三个时间范围比较匹配配置：平均MAE至少改善0.1%，每个范围MSE不升，各段MAE不恶化超过0.25%。合格行情还需通过相对E0的各段MAE与MSE上限。未通过的改动按原方案保留基线，不能因某一开发总分更低就采用。')
sub('限制预测修正的方式')
para('以输入96日Close标准差为局部尺度，行情修正限制在0.5倍局部尺度内，每个辅助分支限制在0.1倍内。行情价格相对最后值居中；量能使用窗口内中心化，训练损失也按窗口尺度归一化。官方固定Close尺度仍用于报告指标，局部标准差设1e-4下限。')
para('初步时间验证仍出现回退后，另登记六个价格反射补充对照。它们共享原模型权重，消除无条件漂移，其他训练条件保持匹配。此约束并不假定真实市场完全对称；未通过门槛时记录为负结果。')

heading('2 行情阶段的优化结果',True)
rows=[]
for d in market['decisions'][:6]:
 b=record(d['baseline'])['validation'];c=record(d['candidate'])['validation']
 rows.append([d['stage'],d['baseline']+' → '+d['candidate'],fmt(b['mae']),fmt(c['mae']),status(d['passed'])])
table(['阶段','配置对照','基线开发MAE','改动开发MAE','跨时间门槛'],rows,[.5,2.6,1.25,1.25,1.5])
para('E1比较MAE和Huber；E2缩小GRU隐藏层；E3提高通道dropout；E4整体缩小PatchTST容量；E5减少iTransformer层数；E6比较窗口归一化后的原始和平稳化表示。E4是容量策略的整体比较，不把效果归因于某一个超参数。')
sub('消除无条件漂移的补充对照')
rows=[]
for d in market['decisions'][6:]:
 c=record(d['candidate'])['validation'];rows.append([d['candidate'],fmt(c['mae']),status(d['passed'])])
table(['补充配置','开发MAE','相对匹配对照的门槛'],rows,[3.9,1.4,1.8])
para(f"按预登记规则选中的行情骨干为{fixed['market']}，seed42行情资格{status(fixed['market_eligible'])}。跨时间结论来自全部三组验证，不等同于每个阶段都获得正收益；epoch0表示该范围未接受训练后的修正。")

heading('3 多模态阶段的优化结果',True)
decisions=modal['decisions']+fixed['stage_E9']
rows=[]
for stage in ('E7','E8','E9','E10'):
 d=next(d for d in decisions if d['stage']==stage)
 b=record(d['baseline'])['validation'];c=record(d['candidate'])['validation']
 rows.append([stage,d['baseline']+' → '+d['candidate'],fmt(b['mae']),fmt(c['mae']),status(d['passed'])])
table(['阶段','配置对照','基线开发MAE','改动开发MAE','验收'],rows,[.5,2.6,1.25,1.25,1.5])
para('E7将96日展开的新闻输出头改为三个固定32日池化段，并重训只有篇数与掩码的对照。E8比较财报数值、财务比率以及匹配掩码年龄。E9比较0.1和0.5的模态dropout。E10将新闻与行情的交叉频谱模式从13降为3，同时比较简单新闻分支。')
sub('内容和结构信息的区别')
for modality,name in [('新闻',modal['news']['id']),('财报',modal['reports']['id'])]:
 r=record(name);key='news' if modality=='新闻' else 'reports';diag=r['diagnostics'].get(key,{})
 para(f"{modality}选定分支为{name}。开发集内容置零的MAE增量为{diag.get('mae_increase',0):+.8f}。该诊断须结合重训覆盖或掩码对照，不能单独证明内容有效。")
para(f"最终固定多模态配置为{fixed['multimodal']}。按seed42跨时间、匹配单模态与内容门槛，整体验收{status(fixed['multimodal_accepted'])}。即使完成导出与推理校验，未通过研究门槛也只能作为实验候选保留。")
para('最终权重审查确认，三个种子的新闻和财报输出头均为0，两个固定候选的预测完全一致。这套多模态配置尚未形成三模态有效依赖，仅作实验归档。',True)
para('财报固定180日延迟，另做270日压力检查；历史版本仍有限。旧比率缓存在滚动A验证范围有72个字段发生不可逆截断，该比率实验仅作近似诊断，详见AUDIT_LIMITATIONS.json。')

heading('4 三种子复验与最终测试',True)
rows=[]
for seed in (42,43,44):
 a=record(fixed['market'],seed=seed)['validation'];b=record(fixed['multimodal'],seed=seed)['validation']
 rows.append([str(seed),fmt(a['mae']),fmt(a['mse']),'与行情完全相同'])
table(['种子','行情开发MAE','行情开发MSE','多模态预测'],rows,[.6,1.7,1.7,3.1])
para('表中是各次训练结果。最终预测固定使用42、43、44等权平均，先平均预测再计算误差；没有挑选分数最好的种子，也没有根据留出分数调整成员或权重。最终固定模型哈希写入TEST_FREEZE.json后才执行测试。')
table(['预测步长','行情留出MAE','行情留出MSE','E0留出MAE','E0留出MSE'],[
 [h,fmt(mr['horizons'][h]['mae']),fmt(mr['horizons'][h]['mse']),fmt(mr['e0']['horizons'][h]['mae']),fmt(mr['e0']['horizons'][h]['mse'])] for h in ('1','6','12','24')],[.6,1.625,1.625,1.625,1.625])
sub('交付与评分校验')
para('两个候选均导出独立model.py与model.pt，默认build_model()严格加载。全开发预测与三个原模型平均一致；每个候选分别用48个原始开发样本去除y后核对预处理与推理，并通过CPU/GPU对齐和辅助模态全缺失检查。')
para('最终内部留出每个固定候选仅评估一次，全部输出有限。重新计算E0与历史结果一致；用PyTorch对保存预测和原始标签独立复算指标，与NumPy差异小于1e-12。测试后未再训练、调系数或改选种子。')

heading('5 当前决定与后续优先级',True)
para('本轮提分目标未达成，两套导出文件均按实验产物保留。最终测试未证明新方案优于基线，多模态输出头归零也不满足有效三模态依赖。当前证据不支持把新包当作已通过验收的正式替代方案。',True)
para('研究效果和课程交付需要同时审查。纯行情候选用于判断价格预测能力；课程原README还要求forward使用行情、新闻和财报三种模态。多模态代码能运行并不代表数值内容已经有益，是否满足有效依赖及内容增量须结合前述诊断。')
para('第一优先级是保留未揭示的时间段：当前内部留出已反复可见，今后只用于描述，不继续围绕它搜索。若继续研究，应先固定新时间范围、训练截止点、标准化规则、基线和验收阈值，再统一比较。')
para('第二优先级是按失败环节优化：行情侧审查局部尺度和各市场阶段的误差；新闻侧保留小容量与覆盖对照；财报侧补真实公告日及历史版本，并从原始记录生成不经过旧截断的比率。每次只扩展有明确证据的分支。')
para('第三优先级是展示与提交整理：报告全部正负结果、固定平均方式、跨时间门槛以及测试边界。未通过验收的配置不写成稳定提升；纯行情模型的成绩不能替代三模态任务要求。')
sub('资源和复现入口')
para('本轮实际训练与最终评估在243完成，主要复用GPU 2、7、4，各卡一个本项目训练进程，独立CUDA依赖和约6GiB显存预算。236保留上一轮独立推理验证环境，230未用于本轮训练。')
para('代码与预登记文件位于experiments/temporal_robust_round2。结果位于其results子目录。PLAN.json与FOLLOWUP_PLAN.json记录两步方案；market_selection.json、modal_selection.json和FINAL_SELECTION.json记录阶段决策；TEST_FREEZE.json和final_holdout/summary.json记录冻结证据及最终评估。',size=10)
para('candidates/market与candidates/multimodal包含独立模型；market_candidate.zip及multimodal_candidate.zip均仅包含model.py和model.pt。全部训练的protocol.json、history.json、result.json和最佳权重均保留，预测与日志保留用于复核。',size=10)

footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=footer.add_run();field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');r._r.addnext(field)
for tree in (doc.element,doc.styles.element):
 for border in list(tree.xpath('.//w:pBdr')):border.getparent().remove(border)
OUT=REPO/'docs/分阶段优化与最终测试复核_2026-11-01.docx'
doc.save(OUT);print(OUT)
