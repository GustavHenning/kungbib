#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# ./get-school-acronyms-and-program-names-data.py
#
# Output: produces a file containing the school acronyms and all of the program names, in the format for inclusion into the thesis template
#
#
# "-t" or "--testing" to enable small tests to be done
# 
# with the option "-v" or "--verbose" you get lots of output - showing in detail the operations of the program
#
# Can also be called with an alternative configuration file:
# ./setup-degree-project-course.py --config config-test.json 1 EECS
#
# Example for a 2nd cycle course for EECS:
#
# ./get-school-acronyms-and-program-names-data.py --config config-test.json
#
# G. Q. Maguire Jr.
#
#
# 2020-02-16
# based on earlier program: get-degree-project-course-data.py
#

import requests, time
import pprint
import optparse
import sys
import json

# Use Python Pandas to create XLSX files
import pandas as pd

from bs4 import BeautifulSoup

################################
######    KOPPS related   ######
################################
KOPPSbaseUrl = 'https://www.kth.se'

English_language_code='en'
Swedish_language_code='sv'

KTH_Schools = {
    'ABE':  ['ABE'],
    'CBH':  ['BIO', 'CBH', 'CHE', 'STH'],
    'EECS': ['CSC', 'EES', 'ICT', 'EECS'], # corresponds to course codes starting with D, E, I, and J
    'ITM':  ['ECE', 'ITM'],
    'SCI':  ['SCI']
}

# https://api.kth.se/api/kopps/v2/schools
#[{"code":"ABE","name":"ABE/Arkitektur och samhällsbyggnad","orgUnit":"A"},{"code":"STH","name":"STH/Teknik och hälsa","orgUnit":"H"},{"code":"ITM","name":"ITM/Industriell teknik och management","orgUnit":"M"},{"code":"BIO","name":"BIO/Bioteknologi","orgUnit":"B"},{"code":"CSC","name":"CSC/Datavetenskap och kommunikation","orgUnit":"D"},{"code":"EES","name":"EES/Elektro- och systemteknik","orgUnit":"E"},{"code":"CHE","name":"CHE/Kemivetenskap","orgUnit":"K"},{"code":"ICT","name":"ICT/Informations- och kommunikationsteknik","orgUnit":"I"},{"code":"XXX","name":"XXX/Samarbete med andra universitet","orgUnit":"U"},{"code":"ECE","name":"ECE/Teknikvetenskaplig kommunikation och lärande","orgUnit":"L"},{"code":"SCI","name":"SCI/Teknikvetenskap","orgUnit":"S"},{"code":"CBH","name":"CBH/Kemi, bioteknologi och hälsa","orgUnit":"C"},{"code":"EECS","name":"EECS/Elektroteknik och datavetenskap","orgUnit":"J"}]
#
# https://api.kth.se/api/kopps/v2/schools?l=en
# [{"code":"ABE","name":"ABE/Architecture and the Built Environment","orgUnit":"A"},{"code":"STH","name":"STH/Technology and Health","orgUnit":"H"},{"code":"ITM","name":"ITM/Industrial Engineering and Management","orgUnit":"M"},{"code":"BIO","name":"BIO/Biotechnology","orgUnit":"B"},{"code":"CSC","name":"CSC/Computer Science and Communication","orgUnit":"D"},{"code":"EES","name":"EES/Electrical Engineering","orgUnit":"E"},{"code":"CHE","name":"CHE/Chemical Science and Engineering","orgUnit":"K"},{"code":"ICT","name":"ICT/Information and Communication Technology","orgUnit":"I"},{"code":"XXX","name":"XXX/Cooperation with other universities","orgUnit":"U"},{"code":"ECE","name":"ECE/Education and Communication in Engineering Science","orgUnit":"L"},{"code":"SCI","name":"SCI/Engineering Sciences","orgUnit":"S"},{"code":"CBH","name":"CBH/Engineering Sciences in Chemistry, Biotechnology and Health","orgUnit":"C"},{"code":"EECS","name":"EECS/Electrical Engineering and Computer Science","orgUnit":"J"}]

def v2_get_schools():
    global Verbose_Flag
    schools_list_swe=[]
    schools_list_eng=[]
    schools=dict()
    old_schools=['BIO', 'CHE', 'ECE', 'CSC', 'EES', 'ICT', 'STH', 'XXX']
    #
    # Use the KOPPS API to get the data
    # note that this returns XML
    url = "{0}/api/kopps/v2/schools".format(KOPPSbaseUrl)
    if Verbose_Flag:
        print("url: " + url)
    #
    r = requests.get(url)
    if Verbose_Flag:
        print("result of getting v2 schools: {}".format(r.text))
    #
    if r.status_code == requests.codes.ok:
        schools_list_swe=r.json()           # simply return the XML
    #
    url = "{0}/api/kopps/v2/schools?l=en".format(KOPPSbaseUrl)
    if Verbose_Flag:
        print("url: " + url)
    #
    r = requests.get(url)
    if Verbose_Flag:
        print("result of getting v2 schools: {}".format(r.text))
    #
    if r.status_code == requests.codes.ok:
        schools_list_eng=r.json()           # simply return the XML
    #
    # Iterate throught the lists and augment the dict
    for s in schools_list_swe:
        name=s['name'].split('/')
        schools[s['code']]={'sv': name[1]}
    #
    for s in schools_list_eng:
        name=s['name'].split('/')
        schools[s['code']]['en']=name[1]
    #
    for s in old_schools:
        schools.pop(s)
    #
    return schools



def v1_get_programmes():
    global Verbose_Flag
    #
    # Use the KOPPS API to get the data
    # note that this returns XML
    url = "{0}/api/kopps/v1/programme".format(KOPPSbaseUrl)
    if Verbose_Flag:
        print("url: " + url)
    #
    r = requests.get(url)
    if Verbose_Flag:
        print("result of getting v1 programme: {}".format(r.text))
    #
    if r.status_code == requests.codes.ok:
        return r.text           # simply return the XML
    #
    return None

def programs_and_owner_and_titles():
    programs=v1_get_programmes()
    xml=BeautifulSoup(programs, "lxml")
    program_and_owner_titles=dict()
    for prog in xml.findAll('programme'):
        if prog.attrs['cancelled'] == 'false':
            owner=prog.owner.string
            titles=prog.findAll('title')
            title_en=''
            title_sv=''
            for t in titles:
                if t.attrs['xml:lang'] == 'en':
                    title_en=t.string
                if t.attrs['xml:lang'] == 'sv':
                    title_sv=t.string
            credits_field=prog.findAll('credits')
            #print("credits_field={}".format(credits_field[0]))
            credit=credits_field[0].string
            program_and_owner_titles[prog.attrs['code']]={'owner': owner, 'title_en': title_en, 'title_sv': title_sv, 'credits': credit}
    #
    return program_and_owner_titles

def main():
    global Verbose_Flag
    parser = optparse.OptionParser()

    parser.add_option('-v', '--verbose',
                      dest="verbose",
                      default=False,
                      action="store_true",
                      help="Print lots of output to stdout"
    )

    options, remainder = parser.parse_args()

    Verbose_Flag=options.verbose
    if Verbose_Flag:
        print("ARGV      : {}".format(sys.argv[1:]))
        print("VERBOSE   : {}".format(options.verbose))
        print("REMAINING : {}".format(remainder))
        print("Configuration file : {}".format(options.config_filename))

    schools=v2_get_schools()
    if Verbose_Flag:
        print("schools={}".format(schools))

    #\newcommand{\schoolAcronym}[1]{%
       #  \ifinswedish
       #  \IfEqCase{#1}{%
       #    {ABE}{\school{Skolan för Arkitektur och samhällsbyggnad}}%
       #    {CBH}{\school{Skolan för Kemi, bioteknologi och hälsa}}%
       #    {EECS}{\school{Skolan för elektroteknik och datavetenskap}}%
       #    {ITM}{\school{Skolan för Industriell teknik och management}}%
       #    {SCI}{\school{Skolan för Teknikvetenskap}}%
       #  }[\typeout{school's code not found}]
       #  \else
       #  \IfEqCase{#1}{%
       #    {ABE}{\school{School of Architecture and the Built Environment}}%
       #    {CBH}{\school{School of Engineering Sciences in Chemistry, Biotechnology and Health}}%
       #    {EECS}{\school{School of Electrical Engineering and Computer Science}}%
       #    {ITM}{\school{School of Industrial Engineering and Management}}%
       #    {SCI}{\school{School of Engineering Sciences}}%
       #  }[\typeout{school's code not found}]
       #  \fi
       #}

    options1=''
    options2=''
    for s in schools:
        st1='    {' + "{}".format(s) + '}{Skolan för ' + "{}".format(schools[s]['sv']) + '}%'
        st2='    {' + "{}".format(s) + '}{School of ' + "{}".format(schools[s]['en']) + '}%'
        options1=options1+st1+'\n'
        options2=options2+st2+'\n'
    #
    cmd='\\newcommand{\\schoolAcronym}[1]{%\n  \\ifinswedish\n  \\IfEqCase{#1}{%\n'+options1
    cmd=cmd+"  }[\\typeout{school's code not found}]\n  \\else\n  \\IfEqCase{#1}{%\n"
    cmd=cmd+options2+"  }[\\typeout{school's code not found}]\n  \\fi\n}\n"
    print("cmd={}".format(cmd))    

    all_programs=programs_and_owner_and_titles()
    if Verbose_Flag:
        print("all_programs={}".format(all_programs))

    options1=''
    options2=''
    for s in all_programs:
        st1='    {' + "{}".format(s) + '}{\programme{' + "{}".format(all_programs[s]['title_sv']) + '}}%'
        st2='    {' + "{}".format(s) + '}{\programme{' + "{}".format(all_programs[s]['title_en']) + '}}%'
        options1=options1+st1+'\n'
        options2=options2+st2+'\n'
    #
    cmdp='\\newcommand{\\programcode}[1]{%\n  \\ifinswedish\n  \\IfEqCase{#1}{%\n'+options1
    cmdp=cmdp+"  }[\\typeout{program's code not found}]\n  \\else\n  \\IfEqCase{#1}{%\n"
    cmdp=cmdp+options2+"  }[\\typeout{program's code not found}]\n  \\fi\n}\n"
    if Verbose_Flag:
        print("cmdp={}".format(cmdp))    


    outpfile_name="schools_and_programs.ins"
    with open(outpfile_name, 'w') as f:
        f.write(cmd)
        f.write(cmdp)
    return

if __name__ == "__main__": main()

