import csv
import re
from doc2vec import make_vector


def get_vector_from_cats():
    with open('/home/ad0915/Desktop/CVFinalDataset/best-artworks-of-all-time/artists.csv') as file:
        read_file = csv.reader(file, delimiter=',')

        name = []
        years = []
        genre = []
        nationality = []
        bio = []

        for row in read_file:
            name.append(row[1])
            years.append(row[2])
            genre.append(row[3])
            nationality.append(row[4])
            bio.append(row[5])

    file.close()

    # Preprocess some lists
    name = [n.replace(' ','_') for n in name[1:]]
    years = [n.replace('–','') for n in years[1:]]
    years = [n.replace('-','') for n in years]
    genre = [n.replace(',', ' ') for n in genre[1:]]
    genre = [n.replace('-', ' ') for n in genre]
    bio = [re.sub('[\W]', ' ', b) for b in bio]
    nationality = nationality[1:]

    cat_features = {'name':name, 'years':years, 'genre':genre, 'bio':bio, 'nationality':nationality}
    print('Done pre-processing features')
    feat_dict = make_vector(cat_features)
    return feat_dict

