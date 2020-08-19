import csv

BGL_sequence=[]
Labels=[]
f=open('Dataset/BGL_parse.csv','r')
reader=csv.DictReader(f)
start=1117784570
seq=[]
label=[]
for line in reader:
    t=float(line['time_stamp'])
    if t-start>300:
        start=t
        if len(seq)>5 and len(seq)<1000:
            BGL_sequence.append(seq)
            Labels.append(label)
        seq=[]
        label=[]
    event_id=int(line['Event_id'])
    if event_id<1 or event_id>394:
        continue
    seq.append(event_id)
    if line['state']=='normal':
        label.append(1)
    else:
        label.append(0)
print('success')
f=open('Dataset/BGL_sessions_tmp.csv','w',newline='')
writer=csv.writer(f)
for line in BGL_sequence:
    writer.writerow(line)
f=open('Dataset/BGL_session_label_tmp.csv','w',newline='')
writer=csv.writer(f)
for line in Labels:
    writer.writerow(line)


f1=open('Dataset/train_data_tmp.csv','w',newline='')
writer1=csv.writer(f1)
f2=open('Dataset/train_data_label_tmp.csv','w',newline='')
writer2=csv.writer(f2)
index=int(len(BGL_sequence)*0.3)
for i in range(index):
    seq=BGL_sequence[i]
    writer1.writerow(seq)
    label=Labels[i]
    writer2.writerow(label)
    # line=[]
    # for j in range(len(seq)):
    #     if label[j]:
    #         line.append(seq[j])
    # writer1.writerow(line)
print('success')




f1=open('Dataset/test_data_session_tmp.csv','w',newline='')
writer1=csv.writer(f1)
f2=open('Dataset/test_data_label_tmp.csv','w',newline='')
writer2=csv.writer(f2)
for i in range(index,len(BGL_sequence)):
    seq=BGL_sequence[i]
    if len(seq)>5:
        label=Labels[i]
        writer1.writerow(seq)
        writer2.writerow(label)
print('success')