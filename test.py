from hanlp_restful import HanLPClient
HanLP = HanLPClient('https://www.hanlp.com/api', auth=None, language='zh') # auth不填则匿名，zh中文，mul多语种

print(HanLP('HanLP为生产环境带来次世代最先进的多语种NLP技术。我的希望是希望张晚霞的背影被晚霞映红。', tasks='tok'))