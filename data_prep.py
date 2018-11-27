import os
import zipfile
import ffmpeg
import datetime

import pyttanko
import utils

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from multiprocessing import Process, Queue, Array

import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine


Base = declarative_base()


class BeatmapMetadata(Base):
    __tablename__ = 'beatmaps'
    # Here we define columns for the table person
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    bmFilePath = Column(String(512), nullable=False)
    audioFilePath = Column(String(512), nullable=False)
    gamemodeType = Column(Integer)
    difficulty = Column(Float)
    dateCreated = Column(DateTime)


class DataPrep:
    def __init__(self):
        self.engine = create_engine(utils.db_path)
        Base.metadata.create_all(self.engine)

        self.beatmaps_root = utils.zipped_beatmaps_dir
        self.extract_root = utils.beatmap_extract_dir

        Base.metadata.bind = self.engine
        self.DBSession = sessionmaker(bind=self.engine)

    def _prep_data_worker(self, start, end, filenames):
        session = self.DBSession()
        i = start
        total_writes = 0

        while i < end:
            print('{}: {}/{}'.format(filenames[i], i - start, end - start))
            extract_path = os.path.join(self.extract_root, filenames[i].replace(" ", "_"))
            os.makedirs(extract_path, exist_ok=True)
            try:
                with zipfile.ZipFile(os.path.join(self.beatmaps_root, filenames[i]), "r") as zip_ref:
                    zip_info = zip_ref.infolist()
                    beatmap_list = []

                    # search for all beatmap files
                    for info in zip_info:
                        if '.osu' in info.filename:
                            # extract beatmap
                            data = zip_ref.read(info)
                            bmfile_path = os.path.join(extract_path, os.path.basename(info.filename))
                            bmfile = open(bmfile_path, 'wb')
                            bmfile.write(data)
                            bmfile.close()
                            # read the beatmap to find related audio file
                            try:
                                file = open(bmfile_path)
                                p = pyttanko.parser()
                                bmap = p.map(file)

                                audio_file_name = bmap.audio_filename.strip().lower()
                                audio_path = os.path.join(extract_path, audio_file_name)
                                wave_filepath = audio_path + '.wav'

                                for jifo in zip_info:
                                    if jifo.filename.lower() == audio_file_name and not os.path.isfile(
                                            os.path.join(extract_path, audio_file_name)):
                                        # extract audio for beatmap
                                        data = zip_ref.read(jifo)
                                        audio = open(audio_path, 'wb')
                                        audio.write(data)
                                        audio.close()
                                        # convert to wav
                                        stream = ffmpeg.input(audio_path)
                                        stream = ffmpeg.output(stream, wave_filepath, ac=1)
                                        ffmpeg.run(stream, quiet=True, overwrite_output=True)
                                # calculate difficulty
                                if bmap.mode == 0:
                                    file = open(bmfile_path)
                                    p = pyttanko.parser()
                                    bmap = p.map(file)
                                    diff = pyttanko.diff_calc().calc(bmap)
                                    file.close()
                                # save metadata to mysql db
                                date = datetime.datetime(*info.date_time[0:6])
                                new_bm_metadata = BeatmapMetadata(bmFilePath=bmfile_path, audioFilePath=wave_filepath,
                                                                  gamemodeType=bmap.mode, difficulty=diff.total,
                                                                  dateCreated=date)
                                session.add(new_bm_metadata)
                            except Exception as e:
                                print("error parsing beatmap or audiofile, deleting beatmap: ", e)
                                os.remove(bmfile_path)
            except(zipfile.BadZipFile):
                print("Bad zipfile: ", filenames[i])
            i += 1
        session.commit()

    def prep_data(self, num_processes):
        filenames = os.listdir(self.beatmaps_root)
        processes = []
        for i in range(num_processes):
            start = i * (len(filenames) // num_processes)
            end = None
            if i != num_processes - 1:
                end = (i + 1) * (len(filenames) // num_processes)
            else:
                end = len(filenames)
            processes.append(Process(target=self._prep_data_worker,
                                     args=(start, end, filenames)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()


def main():
    prep = DataPrep()
    prep.prep_data(8)


if __name__ == "__main__":
    main()
