# case=std_cq -- input=CQ K1ABC FN42
PAYLOAD_std_cq = bytes.fromhex("000000204def1a8a1988")
EXPECTED_std_cq = "CQ K1ABC FN42"
MSGTYPE_std_cq = 6

# case=std_qso -- input=K1ABC W9XYZ FN42
PAYLOAD_std_qso = bytes.fromhex("09bde3506149dc0a1988")
EXPECTED_std_qso = "K1ABC W9XYZ FN42"
MSGTYPE_std_qso = 6

# case=std_report -- input=K1ABC W9XYZ -10
PAYLOAD_std_report = bytes.fromhex("09bde3506149dc1faa48")
EXPECTED_std_report = "K1ABC W9XYZ -10"
MSGTYPE_std_report = 6

# case=std_r_report -- input=K1ABC W9XYZ R-12
PAYLOAD_std_r_report = bytes.fromhex("09bde3506149dc3fa9c8")
EXPECTED_std_r_report = "K1ABC W9XYZ R-12"
MSGTYPE_std_r_report = 6

# case=std_rrr -- input=K1ABC W9XYZ RRR
PAYLOAD_std_rrr = bytes.fromhex("09bde3506149dc1fa488")
EXPECTED_std_rrr = "K1ABC W9XYZ RRR"
MSGTYPE_std_rrr = 6

# case=std_rr73 -- input=K1ABC W9XYZ RR73
PAYLOAD_std_rr73 = bytes.fromhex("09bde3506149dc1fa4c8")
EXPECTED_std_rr73 = "K1ABC W9XYZ RR73"
MSGTYPE_std_rr73 = 6

# case=std_73 -- input=K1ABC W9XYZ 73
PAYLOAD_std_73 = bytes.fromhex("09bde3506149dc1fa508")
EXPECTED_std_73 = "K1ABC W9XYZ 73"
MSGTYPE_std_73 = 6

# case=std_cq_region -- input=CQ DX K1ABC FN42
PAYLOAD_std_cq_region = bytes.fromhex("000046f04def1a8a1988")
EXPECTED_std_cq_region = "CQ DX K1ABC FN42"
MSGTYPE_std_cq_region = 6

# case=std_grid_far -- input=K1ABC W9XYZ RR99
PAYLOAD_std_grid_far = bytes.fromhex("09bde3506149dc1fa3c8")
EXPECTED_std_grid_far = "K1ABC W9XYZ RR99"
MSGTYPE_std_grid_far = 6

# case=free_short -- input=HI
PAYLOAD_free_short = bytes.fromhex("3c470095449f25b00000")
EXPECTED_free_short = "HI"
MSGTYPE_free_short = 0

# case=free_numeric -- input=1234567
PAYLOAD_free_numeric = bytes.fromhex("06c63575ba75bfef0000")
EXPECTED_free_numeric = "1234567"
MSGTYPE_free_numeric = 0

# case=free_punct -- input=TEST/1234
PAYLOAD_free_punct = bytes.fromhex("633895c61fa4d42b2000")
EXPECTED_free_punct = "TEST/1234"
MSGTYPE_free_punct = 0

# case=free_signoff -- input=BYE 73
PAYLOAD_free_signoff = bytes.fromhex("29f3205df238f82c0000")
EXPECTED_free_signoff = "BYE 73"
MSGTYPE_free_signoff = 0

# case=nonstd_compound -- input=PJ4/K1ABC W9XYZ
PAYLOAD_nonstd_compound = bytes.fromhex("0352b0a06149dc1fa448")
EXPECTED_nonstd_compound = "<...> W9XYZ"
MSGTYPE_nonstd_compound = 6
