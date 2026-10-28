"""
Keyword rules for coding "Autre" write-ins and free-text answers to the
survey's own answer options (other_recodes.rule_suggest). The same file in
Niger, Benin and Togo.

Each rule is (name, write-in pattern, answer-option patterns):
  - the write-in pattern is a regex searched in the write-in, folded (lower
    case, no accents, punctuation as spaces; Hausa ƙ ɗ ɓ ƴ as k d b y), in
    Hausa, French or English;
  - the answer-option patterns are regexes searched, in order, in the
    question's own answer labels (folded the same way); the first pattern
    that matches any label gives the code (the shortest matching label).
    A rule whose patterns match no label of a question is not used for it,
    so rules only ever code to options that exist in that survey.
A write-in gets the rule that matches earliest in its text (the first reason
mentioned); on a tie, the rule listed first. Specific rules come before
general ones. Rules in FALLBACK_RULES (catch-alls such as "fear" or "God")
are only used when no other rule matches, so "Tsoron kiba" (fear of weight
gain) is coded to weight gain, not to fear of side effects in general.
Patterns are written in folded form: "don t", not "don't".
"""

FALLBACK_RULES = {"side effects", "religion"}

RULES = [
    # ── Barriers: reasons for not using contraception ─────────────────────────
    ("breastfeeding", r"shayar|\bnono|allait|breastfeed|sevr",
     [r"breastfeed|allait"]),
    ("postpartum", r"haihu kwanan nan|sabuwar haihuwa|\bjego\b|rashin ganin.{0,10}haila|haila.{0,25}\b(dawo|zo)\b|"
                   r"vient d accoucher|retour de couches|recently gave birth|period.{0,10}back|regles.{0,15}(revenu|retour)",
     [r"recently gave birth|menstruation back|retour de couches|vient d accoucher"]),
    ("menopause", r"daina (al ada|haila)|\btsufa|\btsoho|tsohuwa|menopaus|vieill|age avance|old age",
     [r"menopaus"]),
    ("not sexually active", r"rashin karfi|mijina (baya nan|ya yi tafiya|ya tafi|ya rasu|ya mutu)|miji baya nan|"
                            r"bazawara|zawarawa|ba ni da miji|bani da miji|babu miji|\bveuv|divorc|celibat|"
                            r"pas de mari|sans mari|mari (absent|en voyage)|\bexode|\btafiya\b|kauracewa|abstinen|"
                            r"not sexually|no husband|widow|not married",
     [r"not sexually active|sexuellement"]),
    ("husband disapproves", r"miji(na)? (ya ki|baya so|ba ya so|bai yarda|baya yarda|ya hana)|"
                            r"maigida(na)? (ya ki|baya so|ba ya so|bai yarda|ya hana)|"
                            r"mari (refuse|ne veut pas|interdit|n accepte pas|s oppose)|refus du mari|epoux refuse|"
                            r"husband (refuses|disapprove|does not want|doesn t want)",
     [r"husband disapprov|mari (refuse|desapprouve|s oppose)|opposition du (mari|conjoint)"]),
    ("parents disapprove", r"iyaye(na)? (sun ki|ba sa so|basa so|ba su yarda|sun hana)|parents (refusent|refuse|interdi)|"
                           r"my parents",
     [r"parents disapprov|parents (refusent|desapprouvent|s opposent)"]),
    ("spouse's family disapproves", r"suruk|dangin miji|iyalin miji|belle famille|belle mere|beaux parents|in law",
     [r"spouse ?s family|belle famille"]),
    ("siblings disapprove", r"yan uwa(na)? (sun ki|ba sa so|basa so)|freres|soeurs|siblings",
     [r"siblings disapprov|freres.{0,15}(refus|desapprouv|s opposent)"]),
    ("only God decides", r"allah (ne|kadai|shi ke|yake)|ikon allah|nufin allah|dieu (decide|donne)|volonte de dieu|"
                         r"god decides|god s will",
     [r"only god decides|seul dieu"]),
    ("religion does not allow", r"\bharam\b|addini(na)? (bai|ba ya|baya|ya hana)|interdit par (la religion|l islam)|"
                                r"religion (interdit|ne permet)|not allowed",
     [r"does not allow|ne permet pas|religion interdit"]),
    ("religion", r"addini|\ballah\b|ubangiji|religion|\bdieu\b|islam|musulman|coran|kur ani|\bgod\b",
     [r"against religion unspecified|religion.{0,5}(non precise|non specifie|pas precise)", r"contre la religion",
      r"against religion"]),
    ("fear of infertility", r"rashin haihuwa|hana haihuwa|daina haihuwa|ba za ta haihu|ba zan haihu|sterilit|"
                            r"infertil|ne plus (avoir|pouvoir avoir) d enfant|can t have children",
     [r"fear of infertility|sterilit|infertilit"]),
    ("painful period", r"ciwon (ciki|mara).{0,20}haila|haila mai zafi|regles douloureuses|painful period",
     [r"painful period|regles douloureuses"]),
    ("bleeding", r"zubar (da )?jini|\bjini\b|saign|hemorrag|bleed",
     [r"increase in bleeding|saignements abondants|augmentation des saignements", r"bleeding|saign"]),
    ("swelling", r"kumbur|gonfl|swell",
     [r"swelling|gonflement"]),
    ("weight gain", r"\bkiba\b|karuwar jiki|\bteba|grossi|prise de poids|gain de poids|weight gain",
     [r"weight gain|prise de poids"]),
    ("weight loss", r"\brama\b|\bramewa|raguwar jiki|raguewar jiki|maigri|perte de poids|weight loss",
     [r"weight loss|perte de poids|amaigrissement"]),
    ("dizziness / headache", r"ciwon kai|\bjiri\b|\bjuwa\b|vertige|maux? de tete|cephal|headache|dizz",
     [r"dizziness|headache|vertige|maux de tete"]),
    ("cancer", r"ciwon daji|cancer",
     [r"cancer"]),
    ("fear of death", r"tsoron mutuwa|zan mutu|peur de mourir|\bmort\b|\bdeath\b",
     [r"fear of death|peur de (la )?mort|deces"]),
    ("tiredness", r"gajiya|kasala|fatigue|tired",
     [r"tiredness|fatigue"]),
    ("cardiac", r"zuciya|hawan jini|hypertension|tension|coeur|cardiaque|heart|cardiac",
     [r"cardiac|cardiaque|coeur"]),
    ("pimples", r"kuraje|bouton|\bacne|pimple",
     [r"pimples|bouton"]),
    ("dryness", r"bushewa|\bsech|\bdry",
     [r"dryness|secheresse"]),
    ("IUD/implant moving", r"(dasa|karan damtse|implant|sterilet|\bdiu\b|\biud\b).{0,30}(motsa|bace|yawo|bouge|deplac|perdu|mov)",
     [r"iud implant moving|implant.{0,20}(bouge|deplac)"]),
    ("condom lost", r"(kwaroron roba|preservatif|condom).{0,30}(bace|rasa|perdu|reste|lost)",
     [r"condom getting lost|preservatif.{0,20}perdu"]),
    ("reduced pleasure", r"jin dadi|plaisir|pleasure",
     [r"reduced pleasure|plaisir"]),
    ("womb weakness", r"mahaifa|uterus|uterin|womb|matrice",
     [r"womb|uterus|matrice"]),
    ("birth complications", r"matsalolin haihuwa|wahalar haihuwa|complications?.{0,15}accouchement|"
                            r"accouchement difficile|birth complication|cesar|tiyata",
     [r"birth complication|complications? (a l |de l )?accouchement"]),
    ("phobia of drugs", r"allura|wani abu a jiki|piqure|injection|aiguille|medicament|produit chimique|phobi|"
                        r"tsoron magani",
     [r"phobia|phobie|corps etranger"]),
    ("side effects", r"\billa\b|lahani|\bcuta\b|rashin lafiya|matsala|effets? secondaires?|\beffet|consequence|"
                     r"malad|side effect|tsoro|\bpeur\b|\bfear|afraid",
     [r"fear of side effects unspecified|effets secondaires.{0,5}(non precise|non specifie)",
      r"side effect|effets secondaires"]),
    ("too expensive", r"tsada|trop cher|\bcher\b|couteux|expensive",
     [r"too expensive|trop cher|couteux"]),
    ("lack of funding", r"\bkudi\b|kudin|talauci|argent|\bmoyens?\b|pauvre|money|afford|financ",
     [r"lack of funding.{0,50}unspecified|manque de moyens", r"lack of funding|manque d argent"]),
    ("too far", r"\bnisa\b|\bnesa\b|\bloin\b|eloign|distance|\bfar\b",
     [r"too far|trop loin|eloign|distance"]),
    ("not available", r"babu magani|ba a samu|rashin samu|rashin magani|rupture|indisponible|pas disponible|"
                      r"non disponible|stock|not available|unavailable",
     [r"not available|indisponib|rupture|stock|disponib"]),
    ("don't know where/how", r"(\bban san\b|\bbansan\b|ba mu san|ba ta san|ignore|sais pas|connais pas|don t know)"
                             r".{0,25}(\bina\b|inda|\bwuri|\bou\b|where|asibiti|centre)",
     [r"where and how|ou et comment|ou trouver"]),
    ("don't know the methods", r"(\bban san\b|\bbansan\b|sais pas|connais pas|don t know).{0,25}(hanya|methode|method|maganin)",
     [r"method options|methodes disponibles"]),
    ("don't know the benefits", r"(\bban san\b|\bbansan\b|sais pas|connais pas|don t know).{0,25}(amfani|avantage|benefi|utilit)",
     [r"benefits|avantages"]),
    ("lack of knowledge", r"rashin (sani|ilimi|wayewa|fahimta)|jahilci|\bban san\b|\bbansan\b|"
                          r"manque (d |de )(connaissance|information|sensibilisation)|pas (d |assez d )information|"
                          r"ignorance|connais pas|lack of knowledge|no information|not informed|not aware",
     [r"lack of knowledge unspecified|manque de connaissances?.{0,5}(non precise|non specifie)",
      r"lack of knowledge|connaissance"]),
    ("not important / no need", r"(ba|bai) (shi |ya )?da (muhimmanci|mahimmanci)|mahimmanci|muhimmanci|"
                                r"(ba na|bana|ba mu|bamu|ba) bukata|pas besoin|pas necessaire|pas important|inutile|"
                                r"no need|not important|not necessary|don t need",
     [r"not important|pas important"]),
    ("co-wives", r"kishiy|co ?epouse|co wives|polygam",
     [r"co wives|co ?epouses"]),
    ("specific gender", r"(namiji|yaro namiji|yarinya).{0,20}(so|nema|bukata)|(so|nema|bukata).{0,20}(namiji|yarinya)|"
                        r"garcon|\bsexe\b|gender|\bboy\b|\bgirl\b",
     [r"specific gender|sexe"]),
    ("replace a child", r"(yaro|yarona|yarana|\bdan\b|\byar\b).{0,15}(ya mutu|ta mutu|sun mutu|rasu)|remplacer|"
                        r"enfant decede|replace",
     [r"replace a child|remplacer"]),
    ("new couple / first child", r"sabon aure|amarya|\bango\b|ban taba haihuwa|ba ni da (yaro|yara)|bani da (yaro|yara)|"
                                 r"premier enfant|nouveaux? maries?|jeune mariee|pas encore d enfant|first child|"
                                 r"newly married",
     [r"new couple|premier enfant|nouveau couple"]),
    ("want more children", r"(ina|muna|tana|yana|\bta\b|\bya\b) (son|so|bukatar|neman) (karin |kara )?(yara|haihuwa|haihu|haifuwa)|"
                           r"(haihuwa|haifuwa) (nike|nake|muke)|bukatar (samun |karin )?yara|bukatar haihuwa|"
                           r"son haihuwa|kara haihuwa|veux (des|d autres|encore des) enfants|desir d enfants?|"
                           r"vouloir (des )?enfants|encore (des )?enfants|want (more )?(children|kids|to have)",
     [r"not done with childbearing|pas fini", r"desire to have more children unspecified",
      r"desire to have|desir d"]),
    ("body spaces births", r"jiki(na)? (yana|ke|ne).{0,20}tazara|tazara.{0,20}jiki|corps.{0,20}(espace|naturel)|"
                           r"naturellement|body spac",
     [r"body spacing|corps espace"]),
    ("traditional method", r"gargajiya|ganye|saiwa|tsibbu|\blaya\b|traditionn|plante|racine|amulette|gri ?gri|"
                           r"traditional|herb",
     [r"traditional contraception|contraception traditionn|methodes? traditionn"]),
    ("doubts effectiveness", r"(baya|ba ya|bai) aiki|ba sa aiki|basa aiki|shakk|pas efficace|inefficace|ne marche pas|"
                             r"\bdoute|doesn t work|not effective|doubt",
     [r"doubts about contraceptive effectiveness|efficacite"]),
    ("stigma", r"kunya|honte|stigma|ashamed|shame",
     [r"stigma|honte|stigmat"]),
    ("promiscuity", r"yawon banza|karuwanci|\bkaruwa\b|prostitu|infidel|vagabond|libertin|promiscu",
     [r"associating contraceptives with promiscuity|libertin|prostitu"]),
    ("privacy", r"sirri|vie privee|intimite|privacy|\basiri\b",
     [r"privacy|vie privee|intimite"]),

    # ── Drivers: reasons for using contraception ──────────────────────────────
    ("pregnancy outside wedlock", r"(?:(?:guje|gujewa|hana|kauce)\w*\s+)?ciki\s+(?:a\s+)?(?:wajen|bayan|waje)\s+aure|"
                                  r"hors mariage|outside (of )?(marriage|wedlock)|before marriage",
     [r"pregnancy outside of wedlock|grossesse hors mariage"]),
    ("disease outside wedlock", r"cututtuka|\bcuta\b|maladie|infection|\bist\b|\bmst\b|\bsida\b|\bhiv\b|disease",
     [r"avoiding disease|maladies? hors"]),
    ("unwanted pregnancy", r"(guje|gujewa|hana|kauce|kare).{0,15}(ciki|daukar ciki)|ciki (da ba a shirya|ba shiri)|"
                           r"eviter.{0,15}grossesse|grossesse (non desiree|non voulue|non planifi)|unwanted pregnan|"
                           r"avoid pregnan|prevent pregnan",
     [r"avoiding unwanted pregnancy|grossesse non desiree|grossesses? non (voulue|planifi)"]),
    ("husband wants spacing", r"miji(na)? (yana so|ya so|yake so|ya ce)|maigida(na)? (ya so|yana so)|"
                              r"mon mari (veut|souhaite)|husband wants",
     [r"husband wants|mari veut"]),
    ("wife wants spacing", r"(matata|matana|matar) (tana so|ta so|suna so)|ma femme (veut|souhaite)|wife wants|wives want",
     [r"wife my wives want|femme veut"]),
    ("no more children", r"(yara|yarana|haihuwa) (sun isa|ya isa|sun yi yawa)|isassu|(na|mun) gama haihuwa|"
                         r"(ba ma|ba na|bana) son (mu |in )?kara|assez d enfants|suffisamment d enfants|"
                         r"ne (veux|voulons) plus d enfants?|enough children|no more children|don t want more",
     [r"don t want to have more children|ne voulons plus", r"finished giving birth|fini (de|d )",
      r"avoid having too many children|trop d enfants"]),
    ("spacing", r"tazara|espac|spacing|\bspace",
     [r"we want to space|espacer les naissances|space the births"]),
    ("child health", r"lafiyar (yara|yaro|jariri)|sante (des|de l) enfants?|child health|children s health",
     [r"good child health|sante des enfants"]),
    ("mother's strength", r"(karfi|force|strength)",
     [r"increasing strength|force"]),
    ("rest", r"\bhutu|hutawa|\bhuta\b|repos|\brest\b|stress|damuwa",
     [r"more time to rest|temps (de|pour) (se )?repos"]),
    ("food and clothing", r"abinci|sutura|ciyar|nourri|vetement|\bfood|cloth",
     [r"paying for necessities|necessites|nourriture"]),
    ("family finances", r"\bkudi\b|kudin|tattalin arziki|talauci|econom|financ|argent|\bmoyens?\b|pauvret|money|afford",
     [r"reducing economic hardship|difficultes economiques", r"econom|financ"]),
    ("moral upbringing", r"tarbiyy?a|education morale|\bmoral",
     [r"moral upbringing|morale"]),
    ("children's education", r"makaranta|karatu|ilimi|ilmantar|\becole|scolar|educat|school",
     [r"ability to educate|eduquer|scolarit|ecole"]),
    ("time for work", r"\baiki\b|sana a|kasuwanci|travail|carriere|emploi|commerce|\bwork|career|\bjob\b",
     [r"more time for career|carriere|travail"]),
    ("time for religion", r"ibada|\bsalla|prier|priere|\bpray",
     [r"religious practices|pratiques religieuses|prier"]),
    ("care for each child", r"kula da (yara|yarana|kowane)|kulawa|s occuper (des|de chaque) enfants?|care (for|of) (the |each )?child",
     [r"more time to care for each child|s occuper de chaque", r"better care for children"]),
    ("husband promiscuity", r"neman mata|matan waje|infidel|husband promiscu",
     [r"preventing husband promiscuity|infidelite"]),
    ("irregular cycle", r"haila (ba ta|bata) (zuwa )?(daidai|a kan lokaci)|haila ba bisa|irregul|kalanda|calendrier|calendar",
     [r"irregular|irreguli"]),

    # ── Occupation ────────────────────────────────────────────────────────────
    ("farmer", r"\bnoma\b|manomi|\bgona\b|\bkiwo\b|makiyayi|kamun kifi|masunci|agricult|cultivat|paysan|eleveur|"
               r"elevage|pecheur|berger|farm|herd|fish",
     [r"farmer|agricult|cultivat"]),
    ("salesperson", r"kasuwa|ciniki|sayarwa|siyarwa|\bsaida|\bsayar|dillali|\btebur|shago|commer|vend|boutiqu|"
                    r"\bmarch|trad|sell|shop",
     [r"salesperson|small retailer|commer|vend"]),
    ("driver", r"direba|\btuki|\bmota\b|achaba|kabu kabu|okada|chauffeur|\btaxi|\bmoto\b|transport|driver",
     [r"driver|transport|chauffeur"]),
    ("construction", r"\bgini\b|magini|mai gini|\bmacon|batiment|construct|bricklay|mason",
     [r"construction|batiment|macon"]),
    ("artisan", r"dinki|\btela\b|teloli|\bkira\b|makeri|sassaka|kafinta|walda|\baski\b|wanzami|gyaran|makanike|"
                r"mekanik|tailleur|couture|couturi|menuis|soud|forger|coiff|mecani|artisan|tailor|carpent|weld|"
                r"mechanic|barber|craft|sana a|sana ar",
     [r"artisan|craft"]),
    ("housewife", r"matar gida|uwar gida|zaman gida|aikin gida|menager|au foyer|housewife|homemaker",
     [r"housewife|menagere|au foyer"]),
    ("domestic helper", r"aikatau|yar aiki|dan aiki|\bbonne\b|domestique|housemaid|\bmaid\b|domestic",
     [r"domestic helper|domestique"]),
    ("student", r"dalib|makaranta|karatu|almajiri|\beleve|etudiant|student|school|pupil",
     [r"student|eleve|etudiant"]),
    ("religious worker", r"\bliman|limami|malamin addini|malamin islamiyya|alaramma|marabout|\bimam|religieux|pasteur|"
                         r"pretre|religious|cleric",
     [r"religious worker|religieux"]),
    ("health worker", r"asibiti|ungozoma|likita|\bnas\b|nurse|infirm|sage femme|matrone|agent de sante|health worker|"
                      r"doctor|medecin",
     [r"health worker|agent de sante|personnel de sante"]),
    ("traditional medicine", r"\bboka\b|magani(n)? gargajiya|tsibbu|herbor|guerisseur|tradipraticien|traditional (healer|medicine)",
     [r"traditional medicine $|tradipraticien|medecine traditionnelle $"]),
    ("public sector", r"gwamnati|soja|dan sanda|yan sanda|jandarma|police|gendarme|militaire|douane|enseignant|"
                      r"instituteur|malamin makaranta|fonctionnaire|civil servant|government|teacher|soldier",
     [r"public sector|secteur public"]),
    ("private sector", r"kamfani|entreprise|societe|\bprive|\bong\b|\bngo\b|company|private",
     [r"private sector|secteur prive"]),
    ("unemployed", r"babu aiki|ba ni da aiki|bani da aiki|ba aikin yi|zaman banza|\bchom|sans emploi|sans travail|"
                   r"unemploy|jobless|no job",
     [r"unemployed|sans emploi|chom"]),
    ("petty job", r"karamin aiki|kananan ayyuka|lebura|leburori|\bdako\b|petits? (boulot|travail|metier)|journalier|"
                  r"manoeuvre|odd job|petty|labou",
     [r"petty job|petits? (boulot|metier)"]),

    # ── Radio language ────────────────────────────────────────────────────────
    ("Hausa", r"hausa|haoussa", [r"^ hausa $", r"hausa|haoussa"]),
    ("Zarma", r"zarma|djerma|zabarma|songhai", [r"zarma|djerma"]),
    ("Fulfulde", r"fulfulde|fulani|\bfula|\bpeul|bororo|fulatanci", [r"fulfulde|peul"]),
    ("Kanuri", r"kanuri|beriberi|barbarci", [r"kanuri"]),
    ("Tassawaq", r"tass?awaq", [r"tass?awaq"]),
    ("Arabic", r"larabci|arab", [r"arab"]),
    ("French", r"faransanci|francais|french", [r"french|francais"]),
    ("English", r"turanci|anglais|english", [r"english|anglais"]),
]
