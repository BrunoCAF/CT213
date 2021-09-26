from requests import get
import os

def download_file(request_path, file_name):
    with open(file_name, 'wb') as file:
        response = get(request_path)
        file.write(response.content)

subjects = ['fisica','matematica','quimica','portugues','ingles']
years = [str(x) for x in range(2007, 2019)]
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'relative/path/to/file/you/want')

for subject in subjects:
    os.makedirs(dirname + '/Provas/{}'.format(subject))
    for year in years:
        url = 'http://www.vestibular.ita.br/provas/{}_{}.pdf'.format(subject, year)
        file_name = os.path.join(dirname, 'Provas/{}/{}_{}.pdf'.format(subject, subject, year))
        download_file(url, file_name)