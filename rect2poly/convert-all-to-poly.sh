#!/bin/bash

DN_POLY_ID=7
DN_SVD_POLY_ID=11
AB_EX_POLY_ID=12

DN_RECT=dn-2010-2020/dn-2010-2020-rect-json.json
DN_SVD_RECT=dn-svd-2001-2004/dn-svd-2001-2004-rect-json.json
AB_EX_RECT=ab-ex-2001-2004/ab-ex-2001-2004-rect-json.json

python3 main.py --rect $DN_RECT --target_project_id=$DN_POLY_ID > dn-2010-2020/dn-2010-2020-poly-json.json
python3 main.py --rect $DN_SVD_RECT --target_project_id=$DN_SVD_POLY_ID > dn-svd-2001-2004/dn-svd-2001-2004-poly-json.json
python3 main.py --rect $AB_EX_RECT --target_project_id=$AB_EX_POLY_ID > ab-ex-2001-2004/ab-ex-2001-2004-poly-json.json