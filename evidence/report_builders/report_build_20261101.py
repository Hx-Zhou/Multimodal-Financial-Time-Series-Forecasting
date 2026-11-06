from pathlib import Path
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

BASE=Path(__file__).resolve().parent
REPO=BASE/'Multimodal-Financial-Time-Series-Forecasting-main/Multimodal-Financial-Time-Series-Forecasting-main'
PLAN=json.loads((REPO/'experiments/gpu_stage_optimization/PLAN.json').read_text(encoding='utf-8'))
RUN=REPO/'experiments/gpu_stage_optimization'
FINAL=json.loads((RUN/'ensemble/final_summary.json').read_text(encoding='utf-8')) if (RUN/'ensemble/final_summary.json').exists() else None
HOLDOUT=json.loads((RUN/'holdout_20261101/summary.json').read_text(encoding='utf-8')) if (RUN/'holdout_20261101/summary.json').exists() else None
OUT=REPO/'docs/项目进展与分阶段优化计划_2026-11-01.docx'
doc=Document(); sec=doc.sections[0]
sec.page_width=Inches(8.5); sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.7); sec.left_margin=sec.right_margin=Inches(.7)
for name,size in [('Normal',11),('Title',23),('Subtitle',12),('Heading 1',17),('Heading 2',12)]:
 s=doc.styles[name]; s.font.name='Microsoft YaHei'; s.font.size=Pt(size); s.font.color.rgb=RGBColor(0,0,0)
 s.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
 s.paragraph_format.space_after=Pt(7); s.paragraph_format.line_spacing=1.15
 if name.startswith('Heading'): s.paragraph_format.space_before=Pt(10)
doc.core_properties.title='多模态金融预测进展与分阶段优化计划'
doc.core_properties.subject='单人 GPU 优化训练及 MAE MSE 验证'

def para(text,bold=False,size=None):
 p=doc.add_paragraph(); r=p.add_run(text); r.bold=bold
 if size:r.font.size=Pt(size)
 return p
def heading(text,new=False):
 p=doc.add_paragraph(text,'Heading 1'); p.paragraph_format.page_break_before=new; return p
def sub(text):doc.add_paragraph(text,'Heading 2')
def table(headers,rows,widths):
 t=doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
 for cell,w in zip(t.rows[0].cells,widths):cell.width=Inches(w)
 for j,h in enumerate(headers):t.rows[0].cells[j].text=h
 for row in rows:
  cells=t.add_row().cells
  for j,value in enumerate(row):cells[j].text=('0'+str(value)) if str(value).startswith('.') else str(value);cells[j].width=Inches(widths[j])
 for i,row in enumerate(t.rows):
  trpr=row._tr.get_or_add_trPr(); keep=OxmlElement('w:cantSplit');trpr.append(keep)
  if i==0:
   rep=OxmlElement('w:tblHeader');trpr.append(rep)
  for j,cell in enumerate(row.cells):
   cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pr=cell._tc.get_or_add_tcPr();shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'24465C' if i==0 else ('F1F5F7' if i%2==0 else 'FFFFFF'));pr.append(shade)
   margins=OxmlElement('w:tcMar')
   for name,val in [('top','80'),('bottom','80'),('left','100'),('right','100')]:
    el=OxmlElement('w:'+name);el.set(qn('w:w'),val);el.set(qn('w:type'),'dxa');margins.append(el)
   pr.append(margins)
   borders=OxmlElement('w:tcBorders')
   for side in ('top','left','bottom','right'):
    el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
   pr.append(borders)
   for p in cell.paragraphs:
    p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.1
    if widths[j]<1.3:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:
     r.font.name='Microsoft YaHei';r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei');r.font.size=Pt(9.5 if len(headers)>4 else 10)
     r.bold=i==0;r.font.color.rgb=RGBColor(255,255,255) if i==0 else RGBColor(0,0,0)
 spacer=doc.add_paragraph();spacer.paragraph_format.space_after=Pt(0);spacer.paragraph_format.space_before=Pt(0);spacer.paragraph_format.line_spacing=Pt(2);spacer.add_run().font.size=Pt(2)
 return t

doc.add_paragraph('多模态金融预测进展与分阶段优化计划','Title')
doc.add_paragraph('2026年11月1日  单人执行  GPU优化与验证','Subtitle')
if HOLDOUT:
 para('本轮完成40次训练及固定候选测试。最终组合的开发MAE为0.213702，MSE为0.129048；内部留出MAE为0.362970，MSE为0.339169，两项均差于E0和历史模型。开发集提分未泛化到这段留出时间，当前候选尚未通过泛化验证。',True)
 para('内部留出已在此前实验中揭晓，官方隐藏测试仍不可用。测试未训练或调整组合；当前结果不支持宣称新闻及财报结构带来稳定提分。')
elif FINAL:
 para('本轮已完成40次训练配置运行，包括E1至E10的逐阶段对照、9个针对性后续配置及三种子复验。最终固定组合的开发MAE为0.213702，MSE为0.129048；相对历史优化PatchTST分别降低0.61%和0.56%。候选通过预先登记的验收规则，已导出独立提交文件。',True)
 para('这些是反复使用的开发集上的结果，尚不能证明官方隐藏测试集提分。最终模型利用新闻语义与覆盖信息，以及财报缺失掩码和报告年龄；财报数值和比率的独立增量仍未证实。')
else:
 para('当前已完成线性行情基线、多模态扩展和一轮 Transformer 优化。下一步应对每个阶段提出明确改动，通过匹配对照判断是否真正降低 MAE 和 MSE。今晚的目标是形成可复现的优化结果、候选权重与明确的去留结论；是否提分必须由实验决定。',True)
heading('1 任务要求和交付边界')
para('任务用过去96个交易日的行情、新闻和财报，预测未来24个交易日的标准化收盘价。官方最终成绩主要依据隐藏测试集上的MAE和MSE，两项越低越好。[1][2]')
para('project.docx 未限定必须采用 Transformer，也未要求逐项完成 E1至E10。E编号是组内研究路线。原始数据包 README 进一步明确 forward(batch) 使用三种模态，因此应分别保留“指标最优研究候选”和“满足三模态要求的提交候选”；纯行情领先不能直接等同于完整交付。[2][3]')
table(['交付项','验收要求'],[
 ['model.py','build_model()可无参数调用；prepare_batch(samples, device)支持无y样本；forward返回有限的[B,24]结果。'],
 ['model.pt','保存与默认模型完全匹配的state_dict；统计量、固定系数和必要模型成员可随权重重载。'],
 ['独立推理','自写推理逻辑放在model.py；不依赖实验目录绝对路径或测试标签；固定读取器保持兼容。'],
 ['实验与展示','报告总体MAE/MSE、消融、种子波动、预测期限误差以及负结果，展示提分依据和局限。']],[1.15,5.95])
sub('当前最需要解决的问题')
para('开发集与内部留出集的历史排名不一致，局部收益尚未证明跨时间泛化。旧新闻与财报内容通路曾退化，本轮已恢复新闻内容响应，但完整财报仍未超过结构信息对照。根目录model.py继续保留模板；本轮候选包单独保存，不覆盖原实验。')
para('本轮先登记方案再训练，不按已揭晓留出成绩更换结构、种子或系数。所有改动写入独立的experiments/gpu_stage_optimization目录，旧实验保留。')

heading('2 已有进展与历史指标',True)
para('下表均为同一公共切分、标准化Close尺度。除E0外，历史主结果采用固定三种子预测平均；增强E1是等价平均权重单模型。内部留出来自官方验证集，不是官方隐藏测试集。[4–8]')
table(['历史模型','开发MAE','开发MSE','留出MAE','留出MSE'],[
 ['E0 最后值','.217307','.131206','.354286','.330340'],
 ['增强E1 DLinear','.216052','.130342','.351294','.324072'],
 ['E6 平稳化行情','.218733','.132215','.350345','.320201'],
 ['E7 行情加新闻','.217032','.131258','.358696','.328664'],
 ['E8 行情加财报','.218716','.132207','.350379','.320236'],
 ['E9 新闻加财报','.217980','.131718','.352401','.322431'],
 ['E10 频域新闻','.218591','.132141','.350691','.320582'],
 ['优化PatchTST','.215004','.129781','.353688','.328020']],[2.06,1.26,1.26,1.26,1.26])
para('E6在这段内部留出上相对E0降低MAE约1.11%、MSE约3.07%，但开发集弱于增强E1和优化PatchTST。优化PatchTST的开发集结果更好，内部留出却没有超过E6。不能只依据较低留出数字认定最终骨干。[5][8]')
sub('各阶段实际完成程度')
table(['范围','完成内容和剩余问题'],[
 ['E1','原版、中心化、正则化、收缩、三种子与留出评估已完成；收益较小，仍可检验损失与误差尾部。'],
 ['E2与E3','此前因本地空间限制跳过。此次在服务器补齐受控基线，并分别验证池化和通道dropout。'],
 ['E4与E5','PatchTST已完成有限优化和复验；iTransformer此前仅初筛，不能视作架构结论。'],
 ['E6至E10','行情表示、新闻、财报、联合与轻量频域融合均有记录；多模态内容增益尚未确立。'],
 ['后续正则化修复','F1/F2、M1/M2、S1/S2已跑；编码器不再塌缩，但最佳开发检查点均回到epoch0，没有提分。[9]']],[1.35,5.75])
para('特别注意：E9关闭新闻的某次留出数字更低，只是同一模型的诊断，不是独立重训结果，也不能事后选作新赢家。[7]')

heading('3 行情阶段的优化方案',True)
para('本轮基础预算为每配置seed42、最多12轮、patience 3、batch 512。先保留完整训练集，不用缩小样本规模制造不可比结果。以下每个阶段都有基线与针对性改动，未通过门槛则保留基线。[10]')
rows=[]
for s in PLAN['stages'][:6]:rows.append([s['id'],s['baseline'],s['change']+'。'+s['hypothesis']+'。'])
table(['阶段','本轮匹配基线','优化假设和单项改动'],rows,[.55,2.65,3.9])
sub('公平比较规则')
para('配对实验共享数据、种子、初始化规则、优化器、批次顺序和最大训练预算。普通阶段使用MAE+0.2MSE；E1仅改变损失中的MSE系数。E4固定alpha为0.5，并使用相同校准方式的线性对照。早停按实际推理形式计算。')
para('E2选中的池化方式传给E3；E3与E5使用相同原始七变量表示。E6在固定七变量DLinear上比较原始与平稳化表示，因此只能归因为这套骨干上的输入改进。它不会被扩大解释成所有架构都已证明平稳化最优。')
para('本轮统一GPU训练协议与历史CPU方案存在差异，历史成绩只作为水平参照；优化收益以本轮匹配对照计算。不会将不同优化器、预算或单种子与历史三模型平均的差异全部归因于一个结构改动。')

heading('4 多模态阶段的优化方案',True)
para('为承接已有实验并隔离模态增量，E7至E10固定使用本轮E6 stationary权重作为共同受控骨干。即使另一行情模型开发分数更好，也先完成这组对照；将有效分支迁移到更强骨干属于下一项独立实验。')
rows=[]
for s in PLAN['stages'][6:]:rows.append([s['id'],s['change'],s['control']])
table(['阶段','针对性优化','必须比较和检查'],rows,[.55,2.65,3.9])
sub('内容增量需要两层证据')
para('第一层是独立重训：完整新闻须超过只有新闻数量与存在掩码的同结构模型；完整财报须超过只有缺失掩码和报告年龄的匹配模型。已有模型置零的结果不能替代这些重训对照。')
para('第二层是固定权重诊断：语义或财报数值置零、同股票内时间打乱，应使开发MAE恶化至少1e-5，并记录预测变化。该阈值是工程筛选标准；输入敏感、权重范数正常或门控非零，都不足以证明内容有用。')
sub('时间可用性')
para('新闻沿用交易日内去重均值池化，不重新编码512维向量。财报固定采用期末加180个自然日的主假设，270日只做压力检查。缺少真实公告时间和历史版本时，保守延迟仍不能保证完全排除事后修订影响。[6][7]')
para('E9必须超过匹配的新闻、财报单模态及从零联合训练模型；E10必须超过匹配简单新闻融合。失败时记录是哪一项门槛未通过，停止扩张结构。多模态提分若仍未成立，最终提交需诚实保留这一限制。')

heading('5 单人执行顺序与验收标准',True)
table(['资源','计划与实际使用'],[
 ['243','主训练机，8张4090，已有其他任务；选择有显存余量的卡，每卡一个本项目进程，显存预算约6GiB。独立CUDA环境。'],
 ['236','原计划分担训练，NAS读取阻塞后改为本地环境；已完成环境与缓存复制、独立模型GPU推理检查，可供后续使用。'],
 ['230','备用，检查时GPU负载高、根盘余量少，不作为今晚预计完成时间的主要依据。']],[.8,6.3])
table(['顺序','工作与可交付结果'],[
 ['准备','同步必要数据；安装独立CUDA依赖；核对CPU/GPU非零预测、梯度、缺失模态回退；测首轮吞吐量后更新预计耗时。'],
 ['行情','E1/E6、E2/E3/E5、E4按依赖在不同GPU上进行。每个阶段输出匹配对照、逐轮曲线和是否采用改动。'],
 ['多模态','E6 stationary冻结后，新闻和财报两条路线独立训练。随后联合融合；频域路线与简单新闻对照。'],
 ['复验','最多选择两个合格路线补seed43/44及必要匹配对照。若当晚时间不足，明确标记仅单种子初筛，不声称稳定提升。'],
 ['收尾','回传结果及权重；更新本文件；检查可提交模型接口。剩余工作按真实状态列出，训练可从完整epoch恢复。']],[.8,6.3])
sub('采用一项改进的门槛')
para('相对预先指定的匹配对照，开发MAE至少降低0.1%，MSE不升高，较早和较晚开发时段的MAE各不恶化超过0.5%。模态阶段还必须满足内容证据。最多两个最终候选复验三种子，至少2/3方向一致，固定等权平均预测，不挑最好的种子。')
para('开发集共42,930条，边界隔离18,630条，已揭晓内部留出15,390条，训练集587,587条。本轮不再用旧留出选配置。开发时段分组和多种子只能增加稳定性证据，不能恢复独立盲测资格。[4]')
para('基础上限为20个阶段配对配置，加新闻与财报覆盖对照、必要线性参照；不自动扩展网格。每次运行记录协议、父权重哈希、完整曲线、最佳检查点、耗时和停止原因。今晚结束时优先交付完整、可复现的已完成结果。')

heading('6 两轮优化的实际结果',True)
if FINAL:
 para('第一轮23个配置完成E1至E10及必要对照，第二轮9个配置依据已观察到的问题继续验证。随后对固定组合补训两个种子及对应父模型，共8次运行。全部40次均使用完整训练集，训练实际在243完成；236完成独立CUDA环境和最终模型可移植性检查。')
 table(['阶段','验证的改动','结果与决定'],[
  ['E1','提高MSE权重；后续固定残差乘0.5','均回退epoch0，未超过E0；保留历史增强E1参照。'],
  ['E2','最后状态改为平均池化','MAE从0.213691变为0.215064，MSE也恶化，保留最后状态。'],
  ['E3','非Close通道dropout为0.1','MAE 0.216061→0.214825；MSE 0.136561→0.134980。单种子局部通过，未进入最终复验。'],
  ['E4','恢复历史训练校准规则，再缩小前馈层','恢复规则后MAE为0.214783。缩小后为0.214705，但改善仅0.036%，MSE略升，未采用容量改动。'],
  ['E5','两层变量注意力减为一层','两版最优均回退epoch0，未获得升级候选。'],
  ['E6','原始与平稳化表示；后续残差收缩','本轮最优均回退epoch0；不能把历史留出优势当成本轮提分。'],
  ['E7','向量归一化；后续仅取最近20日','完整新闻约0.21394，与重训覆盖对照接近；近期窗口未改善。内容有响应，单分支增量仍弱。'],
  ['E8','MLP改线性；后续改财务比率','比率版MAE为0.214527，但掩码对照为0.214324。财报数值未通过内容增量门槛。'],
  ['E9','独立预训练后联合训练门控和输出头','MAE 0.213254，但早期开发时段与内容检查未通过，未按最低总分直接选用。'],
  ['E10','13个频率模式减为7个','均回退epoch0，未超过简单新闻分支，停止扩大频域结构。']],[.55,2.45,4.1])
 para('第二轮在看到第一轮结果后单独登记，未覆盖原记录。E4第一轮将alpha同时用于训练与推理，与历史有效训练规则不同；修复后重新配对比较，因此最终容量结论以修复后的对照为准。[11]')
 para('E9预训练版本的epoch0已包含单模态训练得到的权重，并非没有学习过。相反，最终组合的三个E6父模型均选中真正的零残差epoch0，行情主干实际为最后收盘价锚点；本轮收益不能归因于DLinear学到了新的历史动态。')
else:para('本节在实际训练结果产生后更新。')

heading('7 开发集候选与后续工作',True)
if FINAL:
 s=FINAL['development'];n=FINAL['matched_news_average']
 para('最终B2组合使用完整新闻、新闻覆盖、财报结构三个成员，按42、43、44三个种子固定等权平均，共9个小模型、596,529个参数。未挑选种子或重调权重。[12]')
 table(['开发集参照','MAE','MSE'],[
  ['E0 最后值','0.217307','0.131206'],
  ['历史增强E1','0.216052','0.130342'],
  ['历史优化PatchTST','0.215004','0.129781'],
  ['本轮匹配新闻三种子平均',f"{n['mae']:.6f}",f"{n['mse']:.6f}"],
  ['最终固定组合',f"{s['mae']:.6f}",f"{s['mse']:.6f}"]],[4.1,1.5,1.5])
 para('相对匹配新闻平均，MAE下降0.174%，MSE下降0.037%；三个种子均改善。早晚开发段MAE分别为0.211214、0.216097，均通过相对E0门槛。表中采用先平均预测再计算误差的口径。')
 para('关闭新闻语义后MAE为0.215061，关闭财报结构分支后为0.214650。组合确实依赖这些输入，但该诊断不能证明财报数值有效，也不能替代独立重训。')
 sub('交付校验已完成')
 para('候选包只含model.py和model.pt。48个官方样本去除y后与训练路径一致，预测最大差异2.38e-7；全开发重载、CPU/GPU对齐、缺失模态及236独立导入推理均通过。')
 sub('下一步优先级')
 if HOLDOUT:
  para('第一优先级是重建时间验证：本次内部留出测试已发现明显退化，当前候选尚未通过泛化验收。后续在训练与开发范围内做多折滚动训练，每折重新计算标准化；在未揭示的时段保留最终评估。已知留出分数不能继续用于搜索组合权重。')
 else:
  para('第一优先级是时间外验证：固定候选、增强E1和PatchTST，在未用于选择的时段比较。历史滚动验证需从截止点前重新训练，并重算该折标准化；旧权重见过的样本不能称为盲测。')
 para('第二优先级是财报可用性与内容：补真实公告日和历史版本后，再比较同报告期比率、季度变化与掩码年龄对照。现阶段不再根据已知分数缩短180日延迟，也不继续扩大频域模型。')
 para('第三优先级是交付效率：合并可等价合并的线性分支，保留三个非线性新闻编码器；以全开发预测一致为验收，不重新调系数。展示时重点讲清假设、对照、E3的局部收益和最终固定组合的收益来源。')
else:para('候选冻结后检查独立加载、无标签样本、缺失模态、输出尺度及权重重载，再交付。')

if HOLDOUT:
 heading('8 内部留出测试结果',True)
 para('按用户追加要求，对已交付的B2固定组合进行一次完整内部留出评估。共15,390条样本、810只股票、19个预测起始日，起始日期为2024年1月24日至2月27日，目标最晚到3月29日。开发目标最晚到1月23日，两个集合的目标日期不重叠。[13]')
 table(['同一内部留出集','MAE','MSE'],[
  ['E0 最后值','0.354286','0.330340'],
  ['历史增强E1','0.351294','0.324072'],
  ['历史E6三种子平均','0.350345','0.320201'],
  ['历史优化PatchTST','0.353688','0.328020'],
  ['本轮B2固定组合',f"{HOLDOUT['internal_holdout']['mae']:.6f}",f"{HOLDOUT['internal_holdout']['mse']:.6f}"]],[4.1,1.5,1.5])
 para('本轮候选相对E0的MAE和MSE分别恶化2.45%和2.67%；相对历史优化PatchTST恶化2.62%和3.40%；相对历史E6平均恶化3.60%和5.92%。因此，此前开发集相对PatchTST的0.61%和0.56%提升，没有延续到这段时间。')
 sub('评测校验')
 para('评测前登记了固定成员、种子、等权规则和文件哈希，确认与已导出的model.py及model.pt完全一致。直接使用原始固定读取器，调用prepare_batch与forward前移除y；全部预测有限，输出为[15390,24]。仅进行推理，没有更新权重。')
 para('E0重新计算后与历史结果一致，排除了切分或指标口径不一致。已保存全部预测、样本索引、逐预测起始日和逐股票指标；另用PyTorch从原始标签独立复算MAE与MSE，与NumPy结果的差异小于1e-12。历史模型数据来自同一切分的既有记录。')
 sub('结论与使用边界')
 para('当前结果支持“开发集收益没有跨时段泛化”，尚不能区分开发集选择过拟合与市场时段变化的影响。当前候选只能保留为实验产物，尚不能作为已验证更优的最终方案；也不能仅因本次已知留出分数最低，就把历史E6当作新选择结果。')

heading('9 依据与复现入口' if HOLDOUT else '8 依据与复现入口',True)
para('课程文件位于proj；以下实验路径均相对内层代码仓库根目录。')
sources=[
 ('[1]','proj/project.docx','课程预测窗口、数据时间范围与隐藏测试评估要求。'),
 ('[2]','proj/hs300_train_val/README.md','原始三模态接口与仅提交两个文件的要求。'),
 ('[3]','docs/experiment_route.md','E1至E10研究问题、单因素原则与消融安排。'),
 ('[4]','experiments/validation_split.json 和 experiments/README.md','公共开发、隔离和内部留出切分。'),
 ('[5]','experiments/e1_dlinear/HOLDOUT.md；experiments/e1_dlinear/e6_plus/HOLDOUT.md','增强E1和E6/E7历史结果，及留出已揭晓的限制。'),
 ('[6]','experiments/e8_dlinear_reports/RESULTS.md','财报实验、内容诊断和公告日期限制。'),
 ('[7]','experiments/e9_dlinear_multimodal/RESULTS.md；experiments/e10_dlinear_spectf/RESULTS.md','联合融合与轻量频域融合结果及归因边界。'),
 ('[8]','experiments/transformer_optimization_common/RESULTS.md','Transformer有限优化、三种子和历史留出比较。'),
 ('[9]','experiments下e8_dlinear_reports_refine、e9_dlinear_multimodal_refine、e10_dlinear_spectf_refine内的RESULTS.md与EXPERIMENT_PLAN.md','已完成正则化修复与尚未执行的后续假设。'),
 ('[10]','experiments/gpu_stage_optimization/PLAN.json','第一轮训练前冻结方案、预算和验收规则。'),
 ('[11]','experiments/gpu_targeted_followup/PLAN.json','依据第一轮结果登记的9个针对性后续配置。'),
 ('[12]','experiments/gpu_targeted_followup/ENSEMBLE_PLAN.json；experiments/gpu_stage_optimization/ensemble/final_summary.json','固定组合筛选、三种子复验、最终指标与推理校验。')]
if HOLDOUT:
 sources.append(('[13]','experiments/gpu_stage_optimization/holdout_20261101/summary.json','本次留出评分、冻结协议、完整预测与独立指标校验。'))
for number,file,note in sources:
 para(number+' '+file,bold=True,size=10).paragraph_format.space_after=Pt(2)
 para(note,size=10).paragraph_format.space_after=Pt(2 if HOLDOUT else 4)
footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=footer.add_run();field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');r._r.addnext(field)
for tree in (doc.element,doc.styles.element):
 for border in list(tree.xpath('.//w:pBdr')):border.getparent().remove(border)
OUT.parent.mkdir(parents=True,exist_ok=True);doc.save(OUT)
print(OUT)
