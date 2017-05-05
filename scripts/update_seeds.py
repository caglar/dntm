from tempfile import mkstemp
from shutil import move
import os
from os import remove, close


def replace_seed(file_path, subst):
    #Create temp file
    fh, abs_path = mkstemp()

    if not os.path.exists(file_path):
        file(file_path, 'w').close()
    try:
        with open(abs_path, 'w') as new_file:
                with open(file_path, "a") as old_file:
                    for line in old_file:
                        if "seed" in line:
                            new_line = "seed = %s" % subst
                            new_file.write(new_line)
                        else:
                            new_file.write(line)
        close(fh)

        #Remove original file
        remove(file_path)

        #Move new file
        move(abs_path, file_path)
    except IOError:
        with open(file_path, "w") as new_file:
            new_line = "seed = %s" % subst
            new_file.write(new_line)

