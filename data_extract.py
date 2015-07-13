# -*- coding: utf-8 -*-
"""
Data scraping and munging from BoxOfficeMojo (http://www.boxofficemojo.com/) 
and Wikipedia (https://en.wikipedia.org/wiki/Main_Page).

Created on Wed Jun 17 01:17:17 2015
@author: ryantonini
"""
import time
import urllib2 # module to read in HTML
import bs4 # BeautifulSoup: module to parse HTML and XML


def open_url(link, max_attempts=5):
        """Open url and save html contents.  
        
        :param link: webpage url
        :type link: str
        :param max_attempts: max # of attempts at opening webpage if initial request fails
        :type max_attempts: int
        :returns: reference to html source code 
        :rtype: bs4 
        """
        try: 
            x = urllib2.urlopen(link)
            htmlSource = x.read() # read in the html source code
            soup = bs4.BeautifulSoup(htmlSource)
            values = soup.find_all('a', href=True) # remove a-tags with no href
            return values
        except urllib2.HTTPError, e:
            print "ERROR", e    
            if max_attempts != 0:
                time.sleep(5) # if http error, try again 5 seconds later
                a = max_attempts - 1
                open_url(link, max_attempts=a)
            else:
                print "MAX ATTEMPTS REACHED"
        except urllib2.URLError, e:
            print "ERROR", e
        
def get_top100movies(yr):
    """Determine the top 100 movies in a given year based on gross box office.
    
    ***Source of Data: BoxOfficeMojo***

    :param yr: year 
    :type yr: int
    :returns: list of the top 100 movies 
    :rtype: list of str or unicode
    """
    link = "http://www.boxofficemojo.com/yearly/chart/?yr=" + str(yr) + "&view=releasedate&view2=domestic&sort=gross&order=DESC&&p=.htm"
    values = open_url(link)
    movies_list = []
    start_end = False  # remove non-movie tags at beginning /end
    for tag in values:
        # find all a tags that correspond to movies
        if tag.get('href')[0:7] == "/movies":
            if tag.string == "Movies A-Z":
                start_end = not start_end
                continue 
            if start_end:
                movies_list.append(tag.string)       
    return movies_list
      
def get_trainingmovies(yrs):
    """Get all box office movies in the specified years.  

    :param yrs: years 
    :type yrs: list of int
    :returns: movies to be used in the training set 
    :rtype: list of str or unicode
    """
    training_movies = []
    for yr in yrs:
        movies = get_top100movies(yr)
        while (len(movies) != 100):
            movies = get_top100movies(yr)
        training_movies += movies   
    return training_movies
        
def get_sequels(ext):
    """Get all movie sequels.  
    
    ***Source of data: Wikipedia***

    :param ext: url extension to webpage
    :type ext: str
    :returns: list of movie sequels
    :rtype: list of str or unicode
    """
    sequels_list = []
    link = "https://en.wikipedia.org" + ext
    values = open_url(link)
    check = False
    incr = 0  
    for tag in values:
        incr += 1
        if tag.contents != []:
            if check:
                # remove '(film)' if in movie name 
                if "(film)" in tag.contents[0]:
                    sequels_list.append(tag.contents[0][:-7])
                else:
                    sequels_list.append(tag.contents[0])
            # specifies we are ready to start adding items to sequels_list
            if tag.contents[0] == "learn more":
                check = True 
            # recursive step - move to the next webpage and call function again
            if tag.contents[0] == "next page" and incr > 45:
                ext = tag.get("href")
                sequels_list = sequels_list[1:-1] 
                return sequels_list + get_sequels(ext)
            # base case - when the last webpage of movies has been reached, exit
            if (tag.contents[0] == "previous page" and incr > 45 and
                    values[values.index(tag, 45) + 1].contents[0] != "next page"):
                sequels_list = sequels_list[1:-1]
                return sequels_list
                
        
    
    