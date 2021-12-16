
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

colors=["#366F70", "#592475", "#320633", "#104B78", "#387934", "#286A4D"]

WITH_RAW_DATA=False

annotations_file = "/data/gustav/datalab_data/model/dn-2010-2020/test_annotations.json"

IMAGES_FOR_THESIS = ["dark-10120707_part01_page006_TFE3FSf.jpg",
"dark-10120964_part01_page005_FZKQ4Zg.jpg",
"dark-10120964_part01_page017_AVThDFz.jpg",
"dark-10120964_part01_page018_OD046mU.jpg",
"dark-10120964_part01_page032_y3LXnnL.jpg",
"dark-10366628_part01_page002_KZBgVFj.jpg",
"dark-10366628_part01_page021_BSsg1sL.jpg",
"dark-10366628_part02_page017_9rwJ51v.jpg",
"dark-4407198_part03_page015_7Zovctl.jpg",
"dark-4425041_part01_page001_JIefsDa.jpg",
"dark-4425337_part02_page051_GQU6vjW.jpg",
"dark-4993682_part01_page022_zk6UnuL.jpg",
"dark-5027440_part01_page028_Jd55Bvg.jpg",
"dark-6608804_part01_page011_TZqA3El.jpg",
"dark-6733161_part04_page033_GqEyiAA.jpg",
"dark-6745860_part04_page037_1RoKim0.jpg",
"dark-6748921_part04_page009_kvtooQ3.jpg",
"dark-6749981_part02_page031_PuwcAOD.jpg"]

ocr_data_loc = "/data/gustav/datalab_data/model/dn-2010-2020/text/" 

def make_images(with_ocr, with_bbox):
	my_dpi=96
	with open(annotations_file, 'r') as f:
		data = json.load(f)
		#print(data) #data["images"] data["annotations"]
		img_info = pd.json_normalize(data["images"])
		anns = pd.json_normalize(data["annotations"])
		print(img_info)
		print(anns["segmentation"])
		for thesis_image in IMAGES_FOR_THESIS:

			fig, ax = plt.subplots()

			orig_width=img_info[img_info["file_name"] == "images/{}".format(thesis_image)]["width"].iloc[0]
			orig_height=img_info[img_info["file_name"] == "images/{}".format(thesis_image)]["height"].iloc[0]
			img_id=img_info[img_info["file_name"] == "images/{}".format(thesis_image)]["id"].iloc[0]


			im = Image.open("/data/gustav/datalab_data/model/dn-2010-2020/images/" + thesis_image)

			fig = plt.figure(frameon=False)
			fig.set_size_inches(orig_width/my_dpi,orig_height/my_dpi)
			ax = plt.Axes(fig, [0., 0., 1., 1.])
			ax.set_axis_off()
			fig.add_axes(ax)

			fig.set_size_inches(orig_width/my_dpi, orig_height/my_dpi, forward=True)
			if WITH_RAW_DATA:
				ax.imshow(im)
			else:
				im = Image.new("RGB", (orig_width, orig_height))
				ax.imshow(im)


			if with_bbox:
				bbox = anns[anns["image_id"] == img_id][["category_id", "segmentation"]]
				for index, row in bbox.iterrows():
					cords = np.array(row["segmentation"])
					#print(cords)
					#print(cords.shape)
					#print("2, {}".format(int(cords.shape[1] / 2)))
					cords = cords.reshape(int(cords.shape[1] / 2), 2)
					#print(cords)
					#print(type(cords))
					rect = patches.Polygon(cords, linewidth=3, edgecolor="pink", facecolor=colors[row["category_id"]])
					#rect = patches.Rectangle((cords[0], cords[1]), cords[2], cords[3], linewidth=3, edgecolor="pink", facecolor=colors[row["category_id"]])
			
					# Add the patch to the Axes
					ax.add_patch(rect)
			if with_ocr:
				with open(ocr_data_loc + thesis_image.replace(".jpg", ".json"), 'r') as ocr_f:
					ocr = pd.json_normalize(json.load(ocr_f)["content"])
					for index, row in ocr.iterrows():
						rect = patches.Rectangle((row["x"], row["y"]), row["width"], row["height"], linewidth=3, edgecolor='red' if with_bbox else 'black', facecolor='none' if with_bbox else 'red')
				
						# Add the patch to the Axes
						ax.add_patch(rect)
			if with_bbox and with_ocr:
				fig.savefig("ocr_bbox/{}".format(thesis_image.split("_")[-1]))
			elif with_ocr:
				fig.savefig("ocr/{}".format(thesis_image.split("_")[-1]))
			elif with_bbox:
				fig.savefig("bbox/{}".format(thesis_image.split("_")[-1]))


			#plt.show()
make_images(False, True)
make_images(True, True)
make_images(True, False)