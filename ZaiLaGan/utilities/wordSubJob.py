from utilities.wordSub import *
import yaml
import sys

def main():
    text = sys.argv[1]
    
    with open("./config.yml", "r") as config_file_yaml:
      config = yaml.load(config_file_yaml, Loader = yaml.BaseLoader)
    
    wordSub_model = wordSub(config["Model"]["ws_model"], config["Model"]["pos_model"], config["Model"]["w2v_model"], config["Data"]["anti_dict"])
    print (wordSub_model.get_word_subs(text))

if __name__ == "__main__":
    main()