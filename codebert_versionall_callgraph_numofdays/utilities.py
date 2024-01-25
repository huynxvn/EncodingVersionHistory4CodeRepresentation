import os 

class SharedFunction(object):
    def __init__(self, a_content) -> None:
        self.filepath = f"./result.txt"
        self.content = a_content

    def AppendFile(self) -> bool:
        """
        append a new row to a file
        """
        try:
            if os.path.exists(self.filepath):
                print("Result file exists")
            else:
                # file is nor existed, create a new blank file
                print("Result file is not existed, create a new one")
                open(self.filepath, 'w').close()

            with open(self.filepath, 'a') as file:
                # file.write(self.content)
                file.write(self.content+"\n")
                file.close()
            return True
        except:
            print("Error in writing file")
            return False