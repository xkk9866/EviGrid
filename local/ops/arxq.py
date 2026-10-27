import urllib.request, urllib.parse, xml.etree.ElementTree as E, time
ns = {'a': 'http://www.w3.org/2005/Atom', 'x': 'http://arxiv.org/schemas/atom'}
J = ['"Information Processing"', '"Knowledge-Based Systems"', '"Expert Systems with Applications"', '"Neural Networks"',
     '"Information Sciences"', '"Engineering Applications of Artificial Intelligence"']
for j in J:
    q = f'abs:retrieval AND abs:generation AND (co:{j} OR jr:{j})'
    u = 'http://export.arxiv.org/api/query?search_query=' + urllib.parse.quote(q) + '&max_results=30&sortBy=submittedDate'
    x = urllib.request.urlopen(u, timeout=60).read()
    print('##', j)
    for e in E.fromstring(x).findall('a:entry', ns):
        jr = e.find('x:journal_ref', ns); c = e.find('x:comment', ns)
        print(' ', e.find('a:id', ns).text.split('/')[-1], '|', ' '.join(e.find('a:title', ns).text.split())[:95], '|',
              (jr.text if jr is not None else '')[:50], '|', ' '.join((c.text if c is not None else '').split())[:70])
    time.sleep(3)
