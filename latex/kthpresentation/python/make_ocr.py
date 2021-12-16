
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

f = open('content.json')
WITH_RAW_DATA=False
FILL_OCR=False
 
data = json.load(f)

my_dpi=96
page_one = [d for d in data if "https://betalab.kb.se/dark-3689532#1-1" in d["@id"]]
print(page_one)

orig_width=6026
orig_height=7452

im = Image.open('bib13991099_19011228_0_11404B_0001.png')
# Create figure and axes
fig, ax = plt.subplots()

fig = plt.figure(frameon=False)
fig.set_size_inches(orig_width/my_dpi,orig_height/my_dpi)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

for item in page_one:
	print(item)
	rect = patches.Rectangle((item["box"][0], item["box"][1]), item["box"][2], item["box"][3], linewidth=5, edgecolor='r', facecolor='red' if FILL_OCR else 'none')

	# Add the patch to the Axes
	ax.add_patch(rect)

fig.set_size_inches(orig_width/my_dpi, orig_height/my_dpi, forward=True)

if WITH_RAW_DATA:
	ax.imshow(im)
	fig.savefig("output.png")
else:
	im = Image.new("RGB", (orig_width, orig_height))
	ax.imshow(im)
	fig.savefig("output-black-{}.png".format(FILL_OCR))
