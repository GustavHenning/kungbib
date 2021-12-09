python3 tools/test.py checkpoints/custom/tf/bert_384_all-mpnet-base-v2-tf_1/kungbib-cascade-mask-tf.py checkpoints/custom/tf/bert_384_all-mpnet-base-v2-tf_1/latest.pth --work-dir=checkpoints/custom/tf/bert_384_all-mpnet-base-v2-tf_1 --eval segm bbox --show-dir checkpoints/custom/tf/bert_384_all-mpnet-base-v2-tf_1/analysis --show-score-thr=0.8 --cfg-options data.test.img_prefix='/data/gustav/datalab_data/model/dn-2010-2020/' data.test.ann_file='/data/gustav/datalab_data/model/dn-2010-2020//test_annotations.json'

python3 tools/test.py checkpoints/custom/tf/ratio/run_1/vanilla_1_0.75/kungbib-cascade-mask.py checkpoints/custom/tf/ratio/run_1/vanilla_1_0.75/latest.pth --work-dir=checkpoints/custom/tf/ratio/run_1/vanilla_1_0.75 --eval segm bbox --show-dir checkpoints/custom/tf/ratio/run_1/vanilla_1_0.75/analysis --show-score-thr=0.8 --cfg-options data.test.img_prefix='/data/gustav/datalab_data/model/dn-2010-2020/' data.test.ann_file='/data/gustav/datalab_data/model/dn-2010-2020//test_annotations.json'


Visual descriptions:

dark-10120707_part01_page006_TFE3FSf.jpg

Only news articles. 

Left column: First 3 separate articles, then two smaller divided 50%. Lastly one wide
Right: One big article

dark-10120964_part01_page005_FZKQ4Zg.jpg

Debate article: One big article and to the right a small protruding element of 2 quotations from other articles on the same topic.

dark-10120964_part01_page017_AVThDFz.jpg

News articles. One big top left, one big bottom left. Right column: 5 news articles, 2nd has image with caption.

dark-10120964_part01_page018_OD046mU.jpg

One big news article related image of cars in a big traffic jam/crash. Left page of a multi page article.

dark-10120964_part01_page032_y3LXnnL.jpg

Game page: top left cross word. top right: "dagens bryderi". middle right: Röda kvarn. Bottom: 3 sodoku

dark-10366628_part01_page002_KZBgVFj.jpg

Left: Large article with image. Right column: Other article with image of author between title and text. Bottom: Footer with DN logo

dark-10366628_part01_page021_BSsg1sL.jpg

Left: Big news article with image and caption on top. Bottom left: News article with image on middle bottom. Right Column: 6 news articles.

dark-10366628_part02_page017_9rwJ51v.jpg

Radio program listings: Top left: 3 column wide SR P1 radio table with image in middle top column. Bottom left: 1 column wide tables with column headers after each column. Right: Large advertisement.

dark-4407198_part03_page015_7Zovctl.jpg

Densely packed advertisements. Cinema listings and movie advertisements.

dark-4425041_part01_page001_JIefsDa.jpg

Front page. Logo on top. Three article teasers in boxes, right one has a head protruding above. Left: Big article teaser with image, caption, title, dotted summary, and text. Right: 4 news article teasers, 2nd has image below text. Bottom: Advertisement.

dark-4425337_part02_page051_GQU6vjW.jpg

Page full of obituaries. Same column with except for top right that has 2x column width.

dark-4993682_part01_page022_zk6UnuL.jpg

Fund listings page. No column headers, but sporadically inserted logos of companies across all columns.

dark-5027440_part01_page028_Jd55Bvg.jpg

Top: Weather data. Bottom left: advertisement. Bottom right: advertisement for newspaper.

dark-6608804_part01_page011_TZqA3El.jpg

Full page article. Left: 3 images with captions relating to article. Right: News article with two columns of text. Before text author and image of author.

dark-6733161_part04_page033_GqEyiAA.jpg

Full page ad of apartment for sale. Company logo on top, one big image, three smaller images. Description of apartment.

dark-6745860_part04_page037_1RoKim0.jpg

Image across 100% of the page. Left page of a multi page article.

dark-6748921_part04_page009_kvtooQ3.jpg

Top: Listings of cultural events going on in stockholm. Top image highlight of event with image and text below. Bottom: Diverse advertisements.

dark-6749981_part02_page031_PuwcAOD.jpg

Results listings of different types of spots. Column headers are sports. Top middle has image with caption of person doing sports. Column headers appear after previous text ends.

 

