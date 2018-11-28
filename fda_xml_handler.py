
import xml.sax
import data_structure

class FdaXmlHandler( xml.sax.ContentHandler ):

    def __init__(self):
        self.currentTag = ""
        self.parentTag = []

        self.sections = []
        self.ignore_regions = []
        self.mentions = []

    def startDocument(self):
        pass

    def endDocument(self):
        self.currentTag = ""
        self.parentTag = []

    def startElement(self, tag, attributes):
        if self.currentTag != '':
            self.parentTag.append(self.currentTag)

        self.currentTag = tag

        if tag == 'Section':
            section = data_structure.Section()
            section.id = attributes['id']
            section.name = attributes['name']
            self.sections.append(section)
        elif tag == 'IgnoredRegion':
            ignored_region = data_structure.IgnoredRegion()
            ignored_region.name = attributes['name']
            ignored_region.section = attributes['section']
            ignored_region.start = int(attributes['start'])
            ignored_region.end = int(attributes['start'])+int(attributes['len'])
            self.ignore_regions.append(ignored_region)
        elif tag == 'Mention':
            mention = data_structure.Entity()
            mention.id = attributes['id']
            mention.section = attributes['section']
            mention.type = attributes['type']
            splitted_start = attributes['start'].split(',')
            splitted_len = attributes['len'].split(',')
            for i, _ in enumerate(splitted_start):
                mention.spans.append([int(splitted_start[i]), int(splitted_start[i])+int(splitted_len[i])])
            self.mentions.append(mention)

    def endElement(self, tag):
        if len(self.parentTag) != 0:
            self.currentTag = self.parentTag[-1]
            self.parentTag.pop()
        else:
            self.currentTag = ''


    def characters(self, content):

        if self.currentTag == 'Section':
            if self.sections[-1].text is None:
                self.sections[-1].text = content
            else:
                self.sections[-1].text += content


if __name__ == '__main__':
    '/Users/feili/dataset/ADE Eval Shared Resources/ose_xml_training_20181101'