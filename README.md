# ImageNet 1K Bounding Boxes
For some experiments, you might wanna pass only the `background` of imagenet images vs passing only the foreground. Here, I've included the code to extract the meta-data for the bounding box, cleaning up the the downloaded stuff, and then changing ImageNet Loader to support only the images that have box annotations. 



# Restarting from the scratch
Downloading:
First download the data from [https://image-net.org/data/bboxes_annotations.tar.gz](here):
```bash
wget "https://image-net.org/data/bboxes_annotations.tar.gz"
```

Extract the File:
```bash
tar -xvf bboxes_annotations.tar.gz 
```

Extract every subfolder:
```bash
cd bboxes_annotations
ls | grep .tar.gz | while read f ; do tar -xvf "${f}" ; done
```

Convert dataset to JS:
```bash
python read_xml.py 
```

Clean the extra 50GB extracted files:
```bash
rm *.tar.gz
ls | grep "n.*" | while read f ; do rm -rf "${f}"  ; done 
```

Get Indices that have bounding boxes:
```bash
python get_indices.py 
```

Then simply use pass the files `boxes.pt` and `indices.pt` to your `BackgroundForegroundImageNet` constructor
```python 
dataset = BackgroundForegroundImageNet(root='.', download=False, boxes='boxes.pt', indices='indices.pt')
```
