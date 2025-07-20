# ormlite

A very minimal object relational mapper for sqlite3.

## About

### Features

- Support for foreign keys.
- Flexible django-style queries.
- Support for updating existing objects.

### Quirks

- There is no support for subsequent schema migration. Once a `Model` has been
  persisted to the database, it can no longer be altered. However, it is
  possible to change both `Model` and its respective database table *manually*
  and use `Model.connect(..., check_models=False)` to skip the integrity
  check.
- Relations between tables are implemented based on rowid (not on the table's
  primary key) and are not enforced on the database level.

## Example

The python code containing the schema can be found [below](#code).

```python
# Create a new or open an existing database and create all necessary table and index
# definitions.
Model.connect(":memory:")

# Create some Person objects. save() stores each object in the database and commits
# right away.
hamill = Person("nm0000434", "Mark Hamill").save()
ford = Person("nm0000148", "Harrison Ford").save()

# Create some Movie objects in bulk and commit afterwards.
star_wars = Movie("tt0076759", "Star Wars", date(1977, 5, 25)).save(commit=False)
blade_runner = Movie("tt0083658", "Blade Runner", date(1982, 6, 25)).save(False)
empire = Movie("tt0080684", "The Empire Strikes Back").save(False)
Model.commit()

# Create some Cast objects that reference both Person and Movie objects.
Cast(star_wars, hamill).save()
Cast(star_wars, ford).save()
Cast(blade_runner, ford).save()
Cast(empire, hamill).save()
Cast(empire, ford).save()

# Find all Cast objects that reference a Person object with a name like "ford" and
# order the results by the movie released date in descending order.
for cast in Cast.filter(person__name__contains="ford").order_by("-movie.released"):
    print(cast.movie)

<Movie imdb_id='tt0083658' title='Blade Runner' released=datetime.date(1982, 6, 25)>
<Movie imdb_id='tt0076759' title='Star Wars: Episode IV - A New Hope' released=datetime.date(1977, 5, 25)>
<Movie imdb_id='tt0080684' title='Star Wars: Episode V - The Empire Strikes Back' released=None>
```

## Documentation

### Defining the schema

Columns in a `Model` subclass may be defined two ways:

```python
class Foo(Model):

    # name: type
    foo: str

    # name: (type, arg, ...)
    bar: (str, "pk")
```

`arg` may be one of:

- `"null"`: Allow value to be `null` / `None`, the default for columns is `not
  null`.
- `"pk"`: Set this column as the primary key. `"pk"` may be set on more than
  one column to create a composite primary key.
- `"index"`: Create an index on this column. If this is set on more than one
  column, all columns are added to the same index. In order to create separate
  indices use different suffixes, e.g. `"index5"`, `"index-foo"`.
- `"unique"`: Create a unique index on this column. The rules from `"index"`
  also apply here.

### Builtin types

| Python type               | SQL type                  |
| ------------------------- | ------------------------- |
| `str`                     | text                      |
| `int`                     | integer                   |
| `float`                   | real                      |
| `boolean`                 | boolean (integer)         |
| `bytes`                   | blob                      |
| `"json"` or `dict`        | json                      |
| `date`                    | date (text)               |
| `datetime`                | datetime (text)           |
| `Model` subclass          | foreign key               |


<a id="code"></a>
## Code

### Python code

```python
from datetime import date
from ormlite import Model

class Person(Model):
    imdb_id: (str, "pk")
    name: str

class Movie(Model):
    imdb_id: (str, "pk")
    title: (str, "unique")
    released: (date, "null")

class Cast(Model):
    movie: Movie
    person: Person
```

### SQL code

```sql
create table "person" (
    "imdb_id" text not null,
    "name" text not null,
    primary key ("imdb_id")
);

create table "movie" (
    "imdb_id" text not null,
    "title" text not null,
    "released" date null,
    primary key ("imdb_id")
);
create unique index movie_title_index on "movie"("title");

create table "cast" (
    "movie" integer not null,
    "person" integer not null,
    foreign key (movie) references "movie"(rowid),
    foreign key (person) references "person"(rowid)
);
```
