from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json, os, time, random

chrome_driver_path = './chromedriver'
save_directory = 'downloads/scores'
download_extention = 'MusicXML' #Please refer to the text on download popup in musescore
username = 'ViMusic2019'
password = 'viralint@2019'
random_wait_range = [1, 30] #second

class STATE:
    INVALID = -1
    OK = 0
    UNABLE_TO_DOWNLOAD = 1
    INVALID_DIRECTORY = 2

def state_debug_message(self, state):
    if state == STATE.INVALID:
        pass
    elif state == STATE.OK:
        pass
    elif state == STATE.UNABLE_TO_DOWNLOAD:
        pass
    elif state == STATE.INVALID_DIRECTORY:
        pass

class musescore_comm:
    def __init__(self):
        self.main_page = 'http://www.musescore.com'

        self.username = username
        self.password = password

        self.download_extention = download_extention
        self.save_directory = save_directory
        self.chrome_options = webdriver.ChromeOptions()
        self.random_wait_range = random_wait_range   

        self.chrome_options.add_experimental_option("prefs", {
            "download.default_directory": save_directory,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
        })        

        self.chrome_options.add_argument("--window-size=1920x1080")
        self.chrome_options.add_argument("--disable-notifications")
        self.chrome_options.add_argument('--headless')
        #self.chrome_options.add_argument('--no-sandbox') # required when running as root user. otherwise you would get no sandbox errors. 
                
        self.driver = None
        
        self.selected_instruments = []
        self.current_sorting_type = 'Relevance'
        
        self.scores_dict = {}
        
    def show_current_url(self):
        print('Current URL: ' + self.driver.current_url)
    
    def show_current_config(self):
        print('Current selected instruments: ' + '.'.join(self.selected_instruments) if '.'.join(self.selected_instruments) else 'None')
        print('Current sorting type: ' + self.current_sorting_type)

    def save_screenshot(self, name):
        self.driver.save_screenshot(name)
        
    def connect(self):
        print('Initializing chrome binary...')
        self.driver = webdriver.Chrome(executable_path=chrome_driver_path, options=self.chrome_options)
        
        print('Loading page: ' + self.main_page)
        self.driver.get(self.main_page)

        print('Login account ID: ' + self.username + ' - Password: ' + self.password)
        button = self.driver.find_element_by_class_name("login")
        button.click()
        username = self.driver.find_element_by_id("edit-name")
        username.send_keys(self.username)
        password = self.driver.find_element_by_id("edit-pass")
        password.send_keys(self.password)
        login = self.driver.find_element_by_id("edit-submit")
        login.click()

        print('Go to main score page')
        self.driver.get(self.main_page + '/sheetmusic')

    def login(self):
        pass
    
    def go_to_url(self, url):
        print('Go to url: ' + url)
        self.driver.get(url)
    
    def reconnect(self):
        self.driver.get('https://musescore.com/sheetmusic')

    def get_current_url(self):
        return self.driver.current_url

    def go_forward(self):
        self.driver.forward()
        
    def go_backward(self):
        self.driver.back()
        
    def get_selected_instruments(self):
        return self.selected_instruments
        
    def query_all_scores_url_in_current_filter(self, limit=None):
        '''Get all the scores in the current filter
            arg:
                limit: Limit the number of scores to get
        '''
        def _check_query_over_limit(current_dict, limit):
            state=False
            if limit is not None:
                if len(current_dict) >= limit:
                    print('Reached scores limit number')
                    state = True
                    
            return state
        
        if limit is not None:
            if limit % 20 != 0:
                print('Limit number must be divived by 20')
                return -1
        
        self.scores_dict = {}
        query_state = -1       
        
        #Query the url in first page
        query_state = self.query_all_scores_url_in_current_page()
            
        #Click next after done querying each page, until the last page
        while self._click_next_button() == 1:
            if _check_query_over_limit(self.scores_dict, limit):
                break
                
            if self.query_all_scores_url_in_current_page() == -1:
                break
                
        return 0
            
    def query_all_scores_url_in_current_page(self):
        query_state = -1
        
        try:
            #Find all scores showing in one page
            scores_title = self.driver.find_elements_by_class_name('score-info__title')
                    
            #Get the hyperlink from each score's title
            for title in scores_title:
                try:
                    tag = title.find_element_by_tag_name('a')
                    name = tag.text
                    url = tag.get_attribute('href')
                    self.scores_dict[name] = url
                    
                    print('Collected score url #' + str(len(self.scores_dict)) + ' with NAME : ' + name +  ' and URL: ' + url)
                    query_state = 1
                except:
                    print('Unable to find hyperlink in the score-s title')
        except:
            print('Unable to find any score')
            
        return query_state
    
    def get_name_by_index(self, index):
        return list(self.scores_dict)[index]
    
    def get_url_by_index(self, index):
        return list(self.scores_dict.values())[index]
    
    def get_available_url_count(self):
        return len(self.scores_dict)

    def save_scores_dict_to_json(self, path):
        self.save_dict(self.scores_dict, path)
    
    def load_scores_dict_from_json(self, path):
        self.scores_dict = self.load_dict(path)

    def save_dict(self, dictionary, path):
        if not os.path.isdir(os.path.dirname(path)):
            os.system('mkdir -p ' + self._wrap_string(os.path.dirname(path)))


        try:
            with open(path, 'w') as fp:
                json.dump(dictionary, fp)
        except:
            print('Cant save dict to json')

    def load_dict(self, path):
        with open(path, 'r') as fp:
            dictionary = json.load(fp)
            return dictionary

    def download_all_avalable_score_svg(self, output_path):
        downloaded_count = 0
        unable_downloaded_count = 0

        fail_to_download_dict = {}
        downloaded_dict = {}

        self.save_directory = output_path

        fail_to_download_dict_path = os.path.join(output_path, 'download_fail_scores.json')
        downloaded_dict_path = os.path.join(output_path, 'downloaded.json')

        if os.path.isfile(fail_to_download_dict_path):
            fail_to_download_dict = self.load_dict(fail_to_download_dict_path)

        if os.path.isfile(downloaded_dict_path):
            downloaded_dict = self.load_dict(downloaded_dict_path)

        for index in range(self.get_available_url_count()):

            '''Check if already failed or downloaded'''
            name = self.get_name_by_index(index)

            try:
                url = fail_to_download_dict[name]
                url = downloaded_dict[name]
                continue
            except:
                pass

            state = self.download_score_svg(self.get_url_by_index(index))
            if state == 1:
                downloaded_count = downloaded_count + 1
                print('Successfully download score')
                downloaded_dict[self.get_name_by_index(index)] = self.get_url_by_index(index)
                self.save_dict(downloaded_dict, downloaded_dict_path)
            else:
                unable_downloaded_count = unable_downloaded_count + 1
                print('Unable to download score')
                fail_to_download_dict[self.get_name_by_index(index)] = self.get_url_by_index(index)
                self.save_dict(fail_to_download_dict, fail_to_download_dict_path)

            print('Counting = Successfull: #' + str(downloaded_count) + ' Failed: #' + str(unable_downloaded_count))
            print('')
            print('===========================================================')
            print('')

    def _random_delay_prevent_block_from_website(self):
        random.seed()

        time_to_wait = round(random.randint(self.random_wait_range[0], self.random_wait_range[1]), 2)
        print('Wait for random time: ' + str(time_to_wait) + 's')
        time.sleep(time_to_wait)

    def _wrap_string(self, string):
        '''
            Wrap input string with 'string'
        '''
        return '"' + string + '"'

    def download_score_svg(self, url):
        download_state = -1
        self.go_to_url(url)

        self._random_delay_prevent_block_from_website()
        # Getting score information
        # score_name = self.driver.find_element_by_css_selector('h1.rdHmC._2OGD_').text
        state, att_dict = self.get_score_info()
        if state == 1:
            print('Getting to get score information')
        else:
            print('Unable to get score information')
            return download_state

        # Find SVG link that store music sheet information
        original_url = ''
        try:
            original_url = self.driver.find_elements_by_tag_name('link')[0].get_attribute('href').split('?')[0]
            print('Found download-able SVG link: ' + original_url)
        except:
            print('Unable to find SVG link')
            return download_state

        try:
            original_keyword = 'score_0'
            number_of_pages = int(att_dict['Pages'])
            score_save_dir = os.path.join(self.save_directory, att_dict['Name'].replace(' ',''))
        except:
            print('Failed to record score info')
            return download_state

        if original_url != '':
            print('File will be saved into: ' + score_save_dir)
            os.system('mkdir -p ' + self._wrap_string(score_save_dir))

            result = 0
            for i in range(number_of_pages):
                result = os.system('wget -nc ' + original_url.replace(original_keyword, 'score_' + str(i)) + ' -P ' + self._wrap_string(score_save_dir))
                self._random_delay_prevent_block_from_website()

                if(result == 0):
                    download_state = 1
                else:
                    break

            self.save_dict(att_dict, os.path.join(score_save_dir, 'info.json'))

        self._random_delay_prevent_block_from_website()
        return download_state

    def download_score_link(self, url):
        state = -1

        self.go_to_url(url)
        state = self._click_download_button()
            
        # Wait until the popup window showed
        state = self._waiting_for_popup("article._2O4bQ.oQvef._1byly")

        # Click to the type of score want to download
        state = self._click_score_extension(self.download_extention)

        return True if state == 1 else False

    def get_score_info(self):
        '''
            Get score's infomation
            return:
                attributes_dict: Store score's infomation
        '''
        # Click show more button to reveal info section
        self._click_show_more_button()

        state = 1

        attributes_dict = {}
        try:
            score_name = self.driver.find_element_by_css_selector('h1.rdHmC._2OGD_').text
            score_link = self.get_current_url()

            # Getting infomation from info section
            # info_section = self.driver.find_element_by_css_selector('div._3T5Ga._1M3ft._307fG')
            attributes_elements = self.driver.find_elements_by_css_selector('div._1ASzI._2OGD_:not(._26ZRZ)')
            
            attributes_list = []
            for elem in attributes_elements:
                attributes_list.append(elem.text)

            attribute_names = attributes_list[0::2]
            attribute_values = attributes_list[1::2]
            
            attributes_dict['Name'] = score_name
            attributes_dict['URL'] = score_link   

            for name, value in zip(attribute_names, attribute_values):
                attributes_dict[name] = value

        except:
            state = -1
 
        return state, attributes_dict

    def _waiting_for_popup(self, css_element, timeout=1):
        '''
            Waiting for popup to show
            arg:
                css_element: tag_name and class name connect with dot (ex: span.class1.class2)
                timeout: Timeout to raise the exception
            return:
                -1: Doesn't show
                 1: Found
        '''
        try:
            element = WebDriverWait(self.driver, 1).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_element))
            )
        except:
            print('Extension popup not showing')
            return -1

        return 1

    def _click_button(self, text, css_element, xpath):
        '''
            Click button with given info
            arg:
                text: button name showed in web page
                css_element: tag_name and class name connect with dot (ex: span.class1.class2)
                xpath: element hierarchy with the one found by css_element (ex: '..' to go to parent)
            return:
                -1: Can't click
                 1: Click successfully
        '''
        button_state = -1
        try:
            buttons_found = self.driver.find_elements(By.CSS_SELECTOR, css_element)
            download_button = None
            for b in buttons_found:
                if b.text == text:
                    download_button = b.find_element_by_xpath(xpath)
                    download_button.click()
                    button_state = 1
                    break
        except:
            pass

        print('Unable to click ' + text + ' button') if button_state == -1 else print('Clicked ' + text + ' button')

        return button_state

    def _click_download_button(self):
        return self._click_button('Download', 'span._3R0py', '..')

    def _click_score_extension(self, ext_type):
        return self._click_button(ext_type, 'span._3R0py', '../..')
    
    def _click_show_more_button(self):
        return self._click_button('Show more', 'span._3R0py', '..')

    def _click_next_button(self):
        ''' Try to find the next button
            return:  
                -1 = Next button doesn't exist
                 0 = Next button is disabled
                 1 = Next button found and clicked to next page
        '''
                
        next_button_state = -1
            
        try:
            #Find next button
            next_button = self.driver.find_element(By.CSS_SELECTOR,"li.pager__item.next");
            next_button_tag = next_button.find_element_by_tag_name('a')
            next_button_url = next_button_tag.get_attribute('href')
            
            self.go_to_url(next_button_url)
            
            next_button_state = 1
        except:
            pass
        try:
            #Find next button
            next_button = self.driver.find_element(By.CSS_SELECTOR,"li.pager__item.next.disabled")
            next_button_state = 0
        except:
            pass
        
        print('Next button doesnt exist in current page' if next_button_state == -1 else ('Next button is disabled' if next_button_state == 0 else 'Next button pressed'))
            
        return next_button_state
    
    def choose_instrument(self, instrument_name):        
        try:
            # Try to find the name in current selected instruments 
            
            # Found => Remove the name from selected instruments
            # Click the relevant button
            index = self.selected_instruments.index(instrument_name)

            print('Unselect instrument from filter:' + self._concat_string_filter(instrument_name))
            status = self._click_button_element('(-) ' + instrument_name)
            
            if status is True:
                self.selected_instruments.pop(index)
            
        except:
            # Not Found => Add the name to selected instruments
            # Click the relevant button
            
            print('Select instrument into filter:' + self._concat_string_filter(instrument_name))  
            status = self._click_button_element(instrument_name)
            
            if status is True:
                self.selected_instruments.append(instrument_name)

        print('Current selected instruments: ' + '.'.join(self.selected_instruments) if '.'.join(self.selected_instruments) else 'None')
  
    def _click_button_element(self, name):
        click_status = False
        
        try:
            self.driver.find_element(By.LINK_TEXT, name).click()
            click_status = True
        except:
            print('There is no button with the name: ' + name)
            
        return click_status
    
    def sorting_by(self, filter_type):
        status = self._click_button_element(filter_type) 

        if status is True:
            self.current_sorting_type = filter_type
        
        print('Current sorting type: ' + self.current_sorting_type)

    def go_to_next_page(self):
        pass
            
    def find_song_genre(self, name):
        pass
    
    def _concat_string_filter(self, string):
        string_changed = ' >> ' + string + ' << '
        return string_changed
         
if __name__== "__main__":
    handler = musescore_comm()
    handler.connect()
    save_path = os.path.join("downloads", "scores", "Piano-Percussion")
    handler.go_to_url("https://musescore.com/sheetmusic?sort=view_count&instruments=3%2C0&parts=2")
    handler.query_all_scores_url_in_current_filter()
    handler.save_scores_dict_to_json(os.path.join(save_path, "scores.json"))
    handler.download_all_avalable_score_svg(save_path)