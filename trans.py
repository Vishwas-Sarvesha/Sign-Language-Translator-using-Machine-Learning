from googletrans import Translator

# This script is intended to translate a text to multiple languages which can be used to improve search results.

text = 'ravi is brother of vivek'


translator = Translator()

#for key, value in destination_languages.items():
res = translator.translate(text, dest='kn').text

print(res)
with open('kan.txt', 'w') as the_file:
                    the_file.write(res)
    



#print(soup.encode("utf-8"))