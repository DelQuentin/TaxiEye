new_dots = [[1,5,10],[1,9,10],[7,8,15],[7,12,15]]
line_dots = []
for ndot in new_dots:
    add = True
    for odot in line_dots:
        if ndot[1]>odot[0] and ndot[1]<odot[2]:
            if ndot[0] > odot[0] and abs(ndot[1]-odot[1])<odot[2]-ndot[0]:
                add = False
                break
            if ndot[0] < odot[0] and abs(ndot[1]-odot[1])<ndot[2]-odot[0]:
                add = False
                break
    if add:
        line_dots.append(ndot)
print(line_dots)