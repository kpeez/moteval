import pytest

from moteval.formats.mot_txt import Track, read_mot, write_mot


def test_mot_txt_round_trip(tmp_path):
    tracks = [
        Track(frame=1, track_id=1, x=10.0, y=10.0, w=20.0, h=20.0, conf=1.0),
        Track(frame=2, track_id=1, x=12.0, y=10.0, w=20.0, h=20.0, conf=0.9),
        Track(frame=1, track_id=2, x=100.0, y=100.0, w=30.0, h=40.0, conf=0.5),
    ]
    path = tmp_path / "seq.txt"
    write_mot(path, tracks)
    reread = read_mot(path)

    expected = sorted(tracks, key=lambda t: (t.frame, t.track_id))
    assert reread == expected


def test_read_mot_defaults_conf_when_absent(tmp_path):
    path = tmp_path / "seq.txt"
    path.write_text("1,1,10,10,20,20\n")
    (row,) = read_mot(path)
    assert row.conf == 1.0


def test_read_mot_malformed_numeric_row_names_file_and_line(tmp_path):
    path = tmp_path / "seq.txt"
    path.write_text("1,1,10,10,20,20\n2,2,x,10,20,20\n")
    with pytest.raises(ValueError) as exc:
        read_mot(path)
    message = str(exc.value)
    assert str(path) in message
    assert ":2" in message
