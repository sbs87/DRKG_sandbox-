"""parse_orhanet_xml.py: transforms xml to csv."""

import xml.etree.ElementTree as ET
import csv

__author__      = "Steve Smith, PhD"
__email__       = "sbs@stevenbsmith.net"
__date__        = "11DEC2020"
__version__     = "0.2"

"""
TODO:
-nested forloops sloppy
-Non hardocde input and output paths
-Make it more abstract so that any xml with arbutrary structure can be read out
"""

def main():

     tree = ET.parse("/Users/stevensmith/Projects/DRKG_sandbox-/en_product1.xml")
     root = tree.getroot()


     # open a file for writing

     parsed_xml = open('/Users/stevensmith/Projects/DRKG_sandbox-/en_product1.csv', 'w')

     # create the csv writer object

     csvwriter = csv.writer(parsed_xml)
     headers = ['disorder_id','disorder_orphacode','disorder_name','gene_id','gene_symbol','gene_name','OMIM_REF']
     csvwriter.writerow(headers)


     ## The XML has the following nested structure between disease (1:m) -> gene (1:m) -> source

     """
        Disorder (id - internal to orhpanew as far as I can tell)
           Orphacode [used to lookup IDs on web-based database)
           Name [human-readable]
           DisorderGeneAssociationList
               DisorderGeneAssociation
                   Gene (id - internal to orhpanet as far as I can tell)
                       Name
                       Symbol
                       ExternalReferenceList
                           ExternalReference (id - internal to orphanet)
                               Source [i.e., OMIM, Ensmbl, etc]
                               Reference [ref id to outside source]

     """

     #TODO the following is sloppy there are too many tested for loops. Fix!
     for disorders in root.findall('DisorderList'):
          output=[]    
          for disorder in disorders.findall('Disorder'):
             disorder_id=disorder.attrib['id']
             disorder_name=disorder.find('Name').text
             disorder_orphacode=disorder.find('OrphaCode').text
             #print("{}\t{}\t{}".format(disorder_id,disorder_name,disorder_orphacode))
             for DisorderGeneAssociationList in disorder.findall('ExternalReferenceList'):
                 for DisorderGeneAssociation in DisorderGeneAssociationList.findall('ExternalReference'):
                    s_r_i={disorder_id:{'source':'','ref':''}}
                    for E in DisorderGeneAssociation:
                         if E.tag=="Source":
                              s_r_i[disorder_id]['source']=E.text
                         elif E.tag=="Reference":
                              s_r_i[disorder_id]['ref']=E.text
                    #print(DisorderGeneAssociation.attrib['id'])
                    #disorder_name=DisorderGeneAssociation.find('Name').text
                    #disorder_orphacode=DisorderGeneAssociation.find('OrphaCode').text
                     #for Gene in DisorderGeneAssociation.findall('Source'):
                      #   print(Gene.text)
                       #  gene_id=Gene.attrib['id']
                        # gene_symbol=Gene.find('Symbol').text  # assumes only one tag called NameSymbol under DisorderGeneAssociation
                         #gene_name=Gene.find('Name').text # assumes only one tag called Name under DisorderGeneAssociation

                         # traverse ExternalReferenceList for each gene's OMIM source
                    omim_source='NOT_FOUND' # initialize in case no OMIM record exists
                    #  for ExternalReferenceList in DisorderGeneAssociation.findall('ExternalReferenceList'):
                    #      for ExternalReference in ExternalReferenceList.findall('ExternalReference'):
                    #           print(ExternalReference)
                    #           ExternalReference_id=ExternalReference.attrib['id']
                                   
                    #                # Next part is tricky/sloppy. XML structure is such that source/ref aren't as key/value.
                    #                # Instead, they are both values to to the ExternalReference ID
                    #                # Store each of these as a dict within the ExternalRefID, then extract later
                    #           s_r_i={ExternalReference_id:{'source':'','ref':''}}
                    #           for E in ExternalReference:
                    #                if E.tag=="Source":
                    #                     s_r_i[ExternalReference_id]['source']=E.text
                    #                elif E.tag=="Reference":
                    #                     s_r_i[ExternalReference_id]['ref']=E.text
                    #                else:
                    #                     print('none') #Should only be two tags under ExternalReference, but just in case
                    #                # Only want OMIM source ID. Note in future, can extract all external ref IDs
                    for source_ref_values in s_r_i.values():
                         if source_ref_values['source']=='MeSH':
                              omim_source=source_ref_values['ref']
                    #      # Write at minium Orphanet code, Gene symbol and OMIM source to file
                    csvwriter.writerow([str(disorder_id),str(disorder_orphacode),disorder_name,omim_source])
                             

if __name__ == "__main__":
    main()
