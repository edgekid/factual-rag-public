import wikipedia
import re
from bs4 import BeautifulSoup
import requests
from time import time

def extract_section_content(content, max_w = -1):
    # Regular expression to match section titles

    section_title_re = re.compile(r'==+ .*? ==+')

    # Remove Additional links/references
    content = content.split('== See also ==')[0]
    content = content.split('== References ==')[0]
    
    # Split the content by section titles
    sections = section_title_re.split(content)
    
    if max_w != -1:
        # Get the first k sentences s.t. k is as large as possible and the text contains no more than 'max_w' words
        
        # Use regex to split text into sentences
        word_count = 0
        done = False
        for i, section in enumerate(sections):
            if done:
                sections[i] = ''
                continue

            sentences = re.split(r'(?<=[.!?]) +', section)
            result = []
            
            for sentence in sentences:
                sentence_word_count = len(sentence.split())
                if word_count + sentence_word_count <= max_w:
                    result.append(sentence)
                    word_count += sentence_word_count
                else:
                    done = True
                    break

            sections[i] = ' '.join(result)

    # Filter out empty sections and join them into a single string
    filtered_content = '\n'.join(section.strip() for section in sections if section.strip())

    return filtered_content

def count_words(text):
    words = re.findall(r'\w+', text)
    return len(words)

def fetch_random_wikipedia_pages():
    wikipedia.set_lang("en")
    r_titles = wikipedia.random(pages = 10)
    pages = []
    for r in r_titles:
        try:
            page = wikipedia.page(r)
            pages.append(page)
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation pages by ignoring them
            pass
        except wikipedia.exceptions.PageError as e:
            # Handle pages that do not exist
            pass
    return pages

def get_wikipage_creation_year(url):
    url = url + '?action=info'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    first_time = soup.find(id='mw-pageinfo-firsttime').find_all('td')[1]

    return first_time.text

# Returns titles of 'how_many' random wiki pages which where created after 'year_min' and before 'year_max'
def get_wiki_pages(how_many, year_min = -1, year_max = -1):
    final_p = set()
    min_w = 600
    while len(final_p) < how_many:
        pages = fetch_random_wikipedia_pages()
        word_counts = [count_words(extract_section_content(p.content)) for p in pages]
        for i, w in enumerate(word_counts):
            if w > min_w:
                creation_date = int(get_wikipage_creation_year(pages[i].url)[-4:])
                if (year_min == -1 or creation_date > year_min) and (year_max == -1 or creation_date < year_max):
                    final_p.add(pages[i].title)

                    if len(final_p) == how_many:
                        break
    
    return final_p

def get_pages(titles, max_w = -1):
    return [extract_section_content(wikipedia.page(t, auto_suggest=False).content, max_w) for t in titles]

def generate_wiki_pages(how_many, year_min = -1, year_max = -1, max_w = -1):
    titles = get_wiki_pages(how_many, year_min, year_max)
    return get_pages(titles, max_w)

