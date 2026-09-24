# Initialize an empty library list
library = []

def add_book(title, author):
    # Create a structured dictionary for the book
    book = {
        "title": title,
        "author": author
    }
    library.append(book)
    print(f"'{title}' has been added to the library.")

def view_library():
    if not library:
        print("The library is currently empty.")
        return
    
    print("\n--- Current Library ---")
    for index, book in enumerate(library, start=1):
        print(f"{index}. {book['title']} by {book['author']}")

# --- Testing the System ---
add_book("The Hobbit", "J.R.R. Tolkien")
add_book("1984", "George Orwell")

view_library()
