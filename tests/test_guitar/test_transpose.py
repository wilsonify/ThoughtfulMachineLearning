import pandas as pd
from music21 import chord


class Song():
    def __init__(self, path):
        self.path = path
        self.capo = 0
        self.lines = []
        self.lyrics = []
        self.chords = []
        self.data = pd.DataFrame(columns=["filepath", "line", "type"])
        self.lines = open(path, "r").readlines()
        self.parse()

    def parse(self):
        count = 0
        for line in self.lines:
            count += 1
            line = line.replace("\n", "")
            if line in ['', " "]:
                continue
            elif line.startswith('[tab]'):
                line = line.replace("[tab] ", "")
                self.chords += [line]
                self.data.loc[count, "line"] = line
                self.data.loc[count, "type"] = "chords"
            else:
                self.lyrics += [line]
                self.data.loc[count, "line"] = line
                self.data.loc[count, "type"] = "not_chords"
        self.data['filepath'] = self.path

    @staticmethod
    def transpose_by_capo(string, capo):
        return chord.Chord([string]).transpose(capo).pitchNames[0]


def test_smoke():
    pass


def test_transpose_note():
    assert Song.transpose_by_capo("C", 1) == 'C#'


def test_song():
    song = Song("../data/Whats Up_capo=0.txt")
    assert song.data.shape == ()
    assert song.chords == []
    assert song.lyrics == []
    assert song.lines == ""


def test_create_dataset():
    import glob
    result = pd.DataFrame()
    for f in glob.glob("/home/thom/chords/**/*.txt", recursive=True):
        song = Song(f)
        result = pd.concat([result, song.data])
    result.to_csv("/home/thom/chords.csv")
