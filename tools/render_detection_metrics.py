"""Render the delivered YOLO test metrics as a readable PNG table."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--font',type=Path,required=True)
    args=parser.parse_args()
    record=json.loads((args.root/'docs/test121_metrics.json').read_text(encoding='utf-8'))
    W,H=2100,1660
    image=Image.new('RGB',(W,H),'#FFFFFF');draw=ImageDraw.Draw(image)
    def font(size): return ImageFont.truetype(str(args.font),size)
    title,sub,body,small=font(60),font(30),font(34),font(27)
    navy,muted,blue='#16324F','#526275','#1263A0'
    draw.text((90,58),'道路资产检测类别与指标',font=title,fill=navy)
    draw.text((92,145),'YOLO26l-737B  |  test121：121张图片，588个标注框',font=sub,fill=muted)
    draw.text((92,194),'固定置信度 ≥ 0.359  ·  同类别 IoU ≥ 0.50  ·  一对一匹配',font=sub,fill=muted)
    draw.rounded_rectangle((90,265,2010,388),radius=16,fill='#EAF3FA')
    summary=f"总体 Micro：  P {record['micro_precision']:.2%}      R {record['micro_recall']:.2%}      F1 {record['micro_f1']:.2%}"
    draw.text((125,284),summary,font=font(36),fill=navy)
    draw.text((125,338),f"15类 Macro F1：{record['macro_f1']:.2%}     |     12小类 Macro F1：{record['small_macro_f1']:.2%}",font=sub,fill=muted)
    y=430;row_h=55
    draw.rectangle((90,y,2010,y+65),fill=navy)
    columns=[(120,'资产类别'),(760,'测试GT'),(1040,'精确率 P'),(1380,'召回率 R'),(1740,'F1')]
    for x,text in columns:draw.text((x,y+12),text,font=body,fill='white')
    y+=65
    names=['标识标牌 BS','反光锥筒 ZT','警示桩 JS','防眩板 FXB','声屏障 SPZ','防撞桶 FZT',
           '水马 SM','护栏 HL','龙门架 LMJ','绿化 LH','路灯 LD','桩号 ZH','监控 JK','交通诱导屏 JT',
           '测速抓拍 CS','防眩网 FXW','信号灯 XHD']
    rows=record['per_class']
    assert len(rows)==17 and sum(r['tp']+r['fn'] for r in rows)==588
    for i,row in enumerate(rows):
        if i%2==0:draw.rectangle((90,y,2010,y+row_h),fill='#F3F6F9')
        gt=row['tp']+row['fn'];draw.text((120,y+7),names[row['class_id']],font=body,fill=navy)
        draw.text((800,y+7),str(gt),font=body,fill=muted)
        for x,key in [(1040,'precision'),(1380,'recall'),(1740,'f1')]:
            text=f'{row[key]:.2%}' if gt else '—'
            draw.text((x,y+7),text,font=body,fill=blue if key=='f1' else navy)
        y+=row_h
    draw.line((90,y,2010,y),fill='#D6DFE8',width=2)
    notes=['P：预测框中正确匹配的比例；R：标注目标中被检出的比例；F1：P与R的调和平均。',
           'Micro汇总全部框；Macro按有GT类别等权平均。12小类不含BS、HL、LH。',
           '“—”表示测试集无该类GT。防撞桶仅2例、信号灯仅1例，需结合样本数看待指标。',
           '本图为静态检测指标，不是分类Accuracy、mAP或视频追踪/道路库存准确率。']
    for line in notes:
        y+=43;draw.text((94,y),line,font=small,fill=muted)
    assert y+45<H
    output=args.root/'docs/assets/yolo26_737b_detection_metrics.png'
    output.parent.mkdir(parents=True,exist_ok=True)
    image.save(output,optimize=True,dpi=(220,220))
    print('FIGURE_READY',output,W,H)

if __name__=='__main__':main()
