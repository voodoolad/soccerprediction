import soccerdata as sd
print("soccerdata", getattr(sd, "__version__", "?"))
for name, cls in [("FBref", sd.FBref), ("Understat", sd.Understat), ("ESPN", sd.ESPN),
                  ("FotMob", sd.FotMob), ("MatchHistory", sd.MatchHistory), ("SofaScore", sd.SofaScore)]:
    try:
        leagues = cls.available_leagues()
        print(f"{name} leagues sample:", leagues[:8])
        inst = cls(leagues=['ENG-Premier League'], seasons=[2024])
        if hasattr(inst, "read_schedule"):
            df = inst.read_schedule()
            print(f"{name} schedule rows:", len(df))
        else:
            print(f"{name} has no read_schedule()")
    except Exception as e:
        print(f"{name}: FAILED -> {e!r}")
