
# coding: utf-8

# In[ ]:

from __future__ import unicode_literals
import requests , json , re , spacy , os ,sys 
from flask import Flask, request, jsonify,current_app
import numpy as np
import pandas as pd
import pdfminer
from collections import Counter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, XMLConverter
from pdfminer.layout import LAParams
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from bs4 import BeautifulSoup as bs
from pdfminer.converter import TextConverter, XMLConverter, HTMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import BytesIO,StringIO
nlp = spacy.load("en_core_web_sm")
from math import sqrt


# In[ ]:

cwd = os.getcwd()
file = open("path.txt","a")
file.write(cwd)
file.close()

def convert_pdf(path, format='text', codec='utf-8', password=''):
    rsrcmgr = PDFResourceManager()
    retstr = BytesIO()
    laparams = LAParams()
    if format == 'text':
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    elif format == 'html':
        device = HTMLConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    elif format == 'xml':
        device = XMLConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    else:
        raise ValueError('provide format, either text, html or xml!')
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    maxpages = 0
    caching = True
    pagenos=set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue().decode()
    text = bs(text, "xml")
    fp.close()
    device.close()
    retstr.close()
    return text


# In[ ]:


def get_outphs(path):
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.psparser import PSLiteral
    from pdfminer.pdftypes import resolve1
    fp = open(path, "rb")
    parser = PDFParser(fp)
    document = PDFDocument(parser)
    pages = dict((page.pageid, pageno) for (pageno,page)
                 in enumerate(PDFPage.create_pages(document)))
    def resolve_dest(dest):
        if isinstance(dest, str):
            dest = resolve1(document.get_dest(dest))
        elif isinstance(dest, PSLiteral):
            dest = resolve1(document.get_dest(dest.name))
        if isinstance(dest, dict):
            dest = dest['D']
        return dest
    toc = []
    for (i1, title, dest, a, structelem) in document.get_outlines():
        toc.append({"i1": i1, "raw_title": title})
    return toc


# In[ ]:


from PyPDF2 import PdfFileReader
def _setup_page_id_to_num(pdf, pages=None, _result=None, _num_pages=None):
    if _result is None:
        _result = {}
    if pages is None:
        _num_pages = []
        pages = pdf.trailer["/Root"].getObject()["/Pages"].getObject()
    t = pages["/Type"]
    if t == "/Pages":
        for page in pages["/Kids"]:
            _result[page.idnum] = len(_num_pages)
            _setup_page_id_to_num(pdf, page.getObject(), _result, _num_pages)
    elif t == "/Page":
        _num_pages.append(1)
    return _result


# In[ ]:


import PyPDF2
frames = []
for foldername,subfolders,files in os.walk(r"./bookmark/"):
    cnt = 1
    for file in files:
        # open the pdf file
        object = PyPDF2.PdfFileReader(os.path.join(foldername,file))
        path = os.path.join(foldername,file)
        #print(path)
        fName = path.split('/')[-1]
        final_fName = fName.split('.')[0]
        df = pd.DataFrame(get_outphs(path))
                
        #next section
        f = open(path,'rb')
        p = PdfFileReader(f)
        # map page ids to page numbers
        pg_id_num_map = _setup_page_id_to_num(p)
        o = p.getOutlines()
        
        #
        page = []
        
        for i in o:
            pg_num = pg_id_num_map[i.page.idnum] + 1
            #print(i['/Title'])
            page.append(pg_num)
    
        df['pageno'] = pd.Series(page)

        df['file_name'] = final_fName
        frames.append(df)
        
dtls = pd.concat(frames)
dtls.loc[:, "title"] = dtls.raw_title.str.replace("\n", " ").str.strip().str.lower()
outphs = dtls.raw_title.tolist()
dtls = dtls.reset_index(drop=True)


# In[ ]:


import PyPDF2
i1ts = [] 
for foldername,subfolders,files in os.walk(r"./bookmark/"):
    cnt = 1
    for file in files:		 	
        path = os.path.join(foldername,file)	 
        r = convert_pdf(path, "xml")
        fName = path.split('/')[-1]
        final_fName = fName.split('.')[0]
        phs = []
        for l in r.find_all("textline"):
            d = dict.fromkeys(["text", "tfxt", "sizes"])
            texts = l.find_all("text")
            text = "".join([x.text for x in texts])#.strip()
            if not text: continue
            d["text"] = text.strip()
            d["rtw"] = text
            d["tfxt"] = [x["font"] for x in texts if x.has_attr("font")]
            d["sizes"] = [round(np.float(x.get("size") or .0)) for x in texts]
            d["pageno"] = int(l.findParent("page")["id"])
            d["bbox"] = l.get("bbox")
            phs.append(d)
            

        
        dphs = pd.DataFrame.from_dict(phs).assign(fqs=lambda x: x["sizes"].apply(lambda y: Counter(y).most_common(1)[0][0])).assign(fqf=lambda x: x["tfxt"].apply(lambda y: Counter(y).most_common(1)[0][0])).assign(text=lambda x: x.rtw.str.replace("\n", " ").str.strip().str.lower()).drop(["tfxt", "sizes"], 1)
        pklFile = final_fName + '.csv'
        dphs.to_csv(pklFile)
        i1ts = dtls.i1.tolist()
 


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer(stop_words="english",use_idf=True,norm='l2')
tfv.fit(dtls['title'])
all_url_tfv = tfv.transform(dtls['title'])
all_url_tfv_df = pd.DataFrame(all_url_tfv.toarray(), columns=tfv.get_feature_names())
doc_tfv = all_url_tfv.toarray()


# In[ ]:


def cosine_sim(u,v):
    return np.dot(u,v) / (sqrt(np.dot(u,u)) * sqrt(np.dot(v,v)))


# In[ ]:


def inxter(q):
    sdx, i1t = dtls[dtls.title.str.startswith(q)].pipe(lambda x: (x.index[0], x.i1.iloc[0]))
    #print sdx,i1t
    ex  = sdx + 1
    imapx = (sdx + ex)
    try:
        for i in range(0, imapx):
            inmx = 0
            inmx += i
            in_xiter = inmx - i
            ([str(in_xiter) for _ in range(10)])
    except:
        print
    #print("i1ts[ex]",i1ts[ex])
    #print("i1t",i1t)
    while i1ts[ex] > i1t:
        print 
        ex += 1
    return dtls.loc[[sdx, ex]][["title", "pageno","file_name"]].values.tolist()



# In[ ]:


import numpy as np
page_tfv =TfidfVectorizer(stop_words="english",use_idf=True,norm='l2')
def get_phs_pg(i1, i2):
    
    pklName = i2[2]+'.csv'
    dphs = pd.read_csv(pklName)
    lom = pd.DataFrame()
    
    def locm(q):
        for _ in range(4):
            lom = dphs[(dphs.pageno == q[1])].copy(deep=True)
            lom = lom[pd.notnull(lom['text'])]
            page_tfv.fit(lom['text'])
            page_all_tfv = page_tfv.transform(lom['text'])
            page_doc_tfv = page_all_tfv.toarray()
            
            q = q[0]
            lom = lom.assign(sc = list(map(lambda x: cosine_sim(page_tfv.transform([q]).toarray()[0],x), page_doc_tfv))).sort_values(["sc"], ascending=[0])
           
            
            if lom.empty:
                q[0] = q[0].rsplit(" ", 1)[0].strip()
            else:
                return lom.index[0]
            
    return dphs.loc[locm(i1):locm(i2)].iloc[:0]


# In[ ]:


jawab = {}
def get_answer(q):
    q = q
    

    ntdf = dtls.assign(sc = list(map(lambda x: cosine_sim(tfv.transform([q]).toarray()[0],x), doc_tfv))).sort_values(["sc"], ascending=[0]).iloc[:10][["title", "sc", "pageno","file_name"]]
    b = inxter(ntdf.iloc[0].title)
    rpages = [(groupname, groupdf) for (groupname, groupdf) in get_phs_pg(*b).groupby("pageno")]
    jawab = {}
    jawab["pc"] = [{"pageno": groupname, "content": "".join(groupdf.rtw.replace("\r\n", "\n"))}
           for (groupname, groupdf) in rpages]
    jawab["q"] = q
    jawab["path"] = b[0][2]+ ".pdf"
    return jawab


# In[ ]:


app = Flask("avnet")


@app.route("/info_retrieval", methods=["POST"])
def getanswer():
    data = request.get_json(force=True)
    jawab = get_answer(data["input_text"]) 
    
    
    return jsonify({
        "filename": jawab.get("path"),
        "final_query": jawab.get("q"),
        "page_content": jawab.get("pc"),
        "isValid": True,
        "other_relevant_pages": None
    })


@app.route('/hi')
def hi():
    
    try:
        return 'Hi -------- World!'
    except Exception as error:
        return str(error)
# In[ ]:


#app.run("0.0.0.0", port = 7000, debug = True, use_reloader = False)

