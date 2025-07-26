#
# ormlite.py - A very minimal object relational mapper for sqlite3.
#
# Copyright (c) 2025, Lars Gustäbel
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import sys
import json
import inspect
import sqlite3

from datetime import date, datetime
from collections import namedtuple


Column = namedtuple("Column", "py_type sql_type attrs")


class OrmLiteError(Exception):
    pass

class UniqueError(OrmLiteError):
    pass

class ForeignKeyError(OrmLiteError):
    pass


sqlite3.register_adapter(date, lambda v: v.isoformat())
sqlite3.register_adapter(datetime, lambda v: v.replace(tzinfo=None).isoformat())

sqlite3.register_converter("date", lambda v: date.fromisoformat(v.decode()))
sqlite3.register_converter("datetime", lambda v: datetime.fromisoformat(v.decode()))
sqlite3.register_converter("json", lambda v: json.loads(v))
sqlite3.register_converter("boolean", lambda v: v == 1)

def extract_datetime(v, attr):
    """SQL function that extracts a single field from a datetime column.
    """
    if v is None:
        return None
    dt = datetime.fromisoformat(v)
    return getattr(dt, attr)

def quote(table_name):
    return f'"{table_name}"'


class Model:
    """Base model class for a single database table <-> object mapping.

       Subclass this class and add annotated class attributes that match the columns in the
       database table:

           class Foo(Model):
               column_a : (int, "pk")
               column_b : str
               column_c : Bar

       NOTE: Schema migrations are not supported which is why these attribute definitions may not
       be altered subsequently. Only new classes may be added. If you plan to change classes and
       their corresponding database definitions *manually*, disable the integrity check with
       `check_models=False`.
    """

    _models = {}
    _columns = {}

    conn = None

    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)

        # Register information about all subclasses of Model.
        table = cls._table_name()
        cls._models[table] = cls
        cls._columns[table] = cls._get_columns()

    #
    # Instance methods.
    #
    def __init__(self, *args, **kwargs):
        self._initialize(args, kwargs, from_db=False)

    def __repr__(self):
        table = self._table_name()
        columns = self._columns[table]
        name = self.__class__.__name__
        values = {n: getattr(self, n, None) for n in columns}
        return f"<{name} {' '.join(f'{k}={v!r}' for k, v in values.items())}>"

    #
    # Database and schema initialization.
    #
    @classmethod
    def connect(cls, database, check_models=True, timeout=5.0, check_same_thread=True, uri=False):
        """Connect all classes derived from Model to an sqlite3 database. Set <check_models> to
           False if you altered one or more classes and their table definitions in the database, to
           omit the integrity check. The <database>, <timeout>, <check_same_thread> and <uri>
           arguments are passed to sqlite3.connect().
        """
        assert cls.conn is None, "Database is already connected"
        cls.conn = sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES, autocommit=False,
                                   timeout=timeout, check_same_thread=check_same_thread, uri=uri)
        cls.conn.row_factory = sqlite3.Row
        cls.conn.create_function("extract_datetime", 2, extract_datetime, deterministic=True)

        # Create all new tables and optionally check if the schema still matches the model
        # defintions.
        for model in cls._models.values():
            model._create_table(check_models)

        cls.conn.commit()

    @classmethod
    def cursor(cls):
        """Return a Cursor object for the current connection.
        """
        return cls.conn.cursor()

    @classmethod
    def commit(cls):
        """Commit pending changes in the current connection.
        """
        cls.conn.commit()

    @classmethod
    def rollback(cls):
        """Rollback pending changes in the current connection.
        """
        cls.conn.rollback()

    @classmethod
    def execute(cls, sql, parameters=()):
        """Call execute() with the given sql and parameters and return a cursor object.
        """
        return cls.conn.execute(sql, parameters)

    @classmethod
    def executemany(cls, sql, parameters):
        """Call executemany() with the given sql and parameters and return a cursor object.
        """
        return cls.conn.executemany(sql, parameters)

    @classmethod
    def executescript(cls, sql_script):
        """Call executescript() with the given sql_script and return a cursor object.
        """
        return cls.conn.executescript(sql_script)

    @classmethod
    def close(cls):
        """Close the database connection.
        """
        assert cls.conn is not None, f"Database is not connected"
        cls.conn.close()
        cls.conn = None

    #
    # SQL generation code.
    #
    @classmethod
    def dump_schema(cls):
        """Return all generated sql statements that constitute the database schema.
        """
        sql = []
        for model in cls._models.values():
            sql.append(model._generate_table_sql() + ";")
            if model.indices:
                sql.append(model._generate_index_sql())
            if hasattr(model, "sql"):
                sql.append(model.sql)
            sql.append("")
        return "\n".join(sql)

    @classmethod
    def _generate_table_sql(cls):
        """Generate sql statements to create the table described by this class's
           attributes and annotations.
        """
        table = cls._table_name()
        columns = cls._columns[table]

        sql = []
        sql.append(f"create table {quote(table)} (")

        primary_keys = []
        foreign_keys = []
        index_keys = {}
        unique_keys = {}
        for name, column in columns.items():
            null = "not null"
            for attr in column.attrs:
                if attr == "null":
                    null = "null"
                elif attr == "pk":
                    primary_keys.append(quote(name))
                elif attr == "fk":
                    foreign_keys.append((name, column.py_type._table_name()))
                elif attr.startswith("index"):
                    index_keys.setdefault(attr.removeprefix("index"), []).append(name)
                elif attr.startswith("unique"):
                    unique_keys.setdefault(attr.removeprefix("unique"), []).append(name)
                else:
                    raise ValueError(f"invalid attribute: {attr}")

            sql.append(f"    {quote(name)} {column.sql_type} {null},")

        if primary_keys:
            sql.append(f"    primary key ({', '.join(primary_keys)}),")

        if foreign_keys:
            for a, b in foreign_keys:
                sql.append(f"    foreign key ({a}) references {quote(b)}(rowid),")

        cls.indices = []
        for columns in index_keys.values():
            cls.indices.append((columns, False))
        for columns in unique_keys.values():
            cls.indices.append((columns, True))

        sql[-1] = sql[-1].rstrip(",")
        sql.append(f")")
        return "\n".join(sql)

    @classmethod
    def _generate_index_sql(cls):
        """Generate sql statements to create the necessary indices.
        """
        table = cls._table_name()
        sql = []
        for columns, unique in cls.indices:
            index_name = f"{table}_{'_'.join(columns)}_index"
            column_names = ", ".join(quote(c) for c in columns)
            sql.append(f"create {'unique ' if unique else ''}index {index_name} "\
                    f"on {quote(table)}({column_names});")
        return "\n".join(sql)

    @classmethod
    def _create_table(cls, check_models):
        """Create the table in the database if it does not yet exist. Fail by default if the table
           exists and is different.
        """
        table = cls._table_name()
        table_sql = cls._generate_table_sql()

        row = cls.execute(
                "select sql from sqlite_master where type = 'table' and tbl_name = ?",
                (cls._table_name(),)).fetchone()

        if row is None:
            cls.executescript(table_sql)

            index_sql = cls._generate_index_sql()
            cls.executescript(index_sql)

            if hasattr(cls, "sql"):
                cls.executescript(cls.sql)

        elif check_models:
            # XXX Fortunately, this check is very simple. The sqlite3 database stores the sql for
            # tables that we add verbatim. Hopefully, this does not break in the future.
            if row[0].lower() != table_sql:
                raise RuntimeError(f"the {cls.__name__} model changed, migration is not supported")

    @classmethod
    def _get_columns(cls):
        """Parse the class's attributes and their annotations.
        """
        columns = {}
        annotations = inspect.get_annotations(cls)
        if not annotations:
            raise ValueError(f"missing annotations in {cls}")

        for name, typedef in annotations.items():
            if name != name.lower():
                raise ValueError(f"no support for non-lowercase attributes: {name!r}")

            if not isinstance(typedef, tuple):
                typedef = (typedef,)

            py_type = typedef[0]
            attrs = set(typedef[1:])

            if inspect.isclass(py_type) and issubclass(py_type, Model):
                sql_type = "integer"
                attrs.add("fk")
            elif py_type is str:
                sql_type = "text"
            elif py_type is int:
                sql_type = "integer"
            elif py_type is float:
                sql_type = "real"
            elif py_type is bool:
                sql_type = "boolean"
            elif py_type is bytes:
                sql_type = "blob"
            elif py_type == "json" or py_type is dict:
                py_type = object
                sql_type = "json"
            elif py_type is date:
                py_type = date
                sql_type = "date"
            elif py_type is datetime:
                py_type = datetime
                sql_type = "datetime"
            else:
                raise TypeError(f"unknown type {py_type!r}")

            columns[name.lower()] = Column(py_type, sql_type, attrs)

        return columns

    #
    # Internal instance methods.
    #
    def _initialize(self, args, kwargs, from_db=False):
        """Initialize a newly constructed instance either from the arguments
            passed to its __init__() or from a database table row.
        """
        table = self._table_name()
        columns = list(self._columns[table].items())

        # First handle all positional arguments.
        args = list(args)
        while args:
            key, column = columns.pop(0)
            value = args.pop(0)
            self._set(column, key, value, from_db)

        # Then handle the keyword arguments.
        columns = dict(columns)
        for key, value in kwargs.items():
            column = columns.get(key)
            if column is None:
                raise ValueError(f"invalid key {key!r}")
            self._set(column, key, value, from_db)

    def _set(self, column, key, value, from_db):
        """Set an instance attribute to a value. Check for null constraints and
           the correct type and fetch referenced foreign objects if necessary.
        """
        if value is None:
            if "null" not in column.attrs:
                raise ValueError(f"invalid {key!r} null-value")
        elif from_db and issubclass(column.py_type, Model):
            # Create the object referenced by the foreign key rowid.
            value = column.py_type.get(rowid=value)
        elif not isinstance(value, column.py_type):
            raise ValueError(f"invalid {key!r} value {value!r}, expected {column.py_type}")

        setattr(self, key, value)

    def _prepare_row(self, update=False, commit=True):
        """Prepare a row for this instance for addition to the database.
        """
        table = self._table_name()
        columns = self._columns[table]

        # Prepare the list of values and convert referenced objects to
        # rowids by saving them if necessary.
        values = []
        for n, c in columns.items():
            v = getattr(self, n, None)
            if isinstance(v, Model):
                # Check if the referenced object has already been saved, and
                # has a rowid assigned to it.
                try:
                    v.rowid
                except AttributeError:
                    v = v.save(commit).rowid
                else:
                    if update:
                        v.update(commit)
                    v = v.rowid
            elif c.sql_type == "json":
                v = json.dumps(v)
            values.append(v)

        return table, columns, values

    def save(self, commit=True):
        """Save this instance as a new row to the database.
        """
        table, columns, values = self._prepare_row(commit=commit)

        columns = ", ".join(quote(c) for c in columns)
        placeholders = ", ".join("?" for v in values)

        try:
            self.rowid = self.execute(f"""
                    insert into {quote(table)} ({columns}) values ({placeholders}) returning rowid
                """, values).fetchone()[0]
        except sqlite3.IntegrityError as exc:
            raise UniqueError(f"{self} exists already") from exc

        if commit:
            self.conn.commit()
        return self

    def update(self, commit=True):
        """Update an existing row with new values from this instance.
        """
        table, columns, values = self._prepare_row(update=True, commit=commit)

        self.execute(f"update {quote(table)} set {', '.join(f'{quote(c)} = ?' for c in columns)} where rowid = ?",
                     values + [self.rowid])
        if commit:
            self.conn.commit()
        return self

    def _find_references(self):
        """Find foreign key references to this instance's model in other models.
        """
        tables = []
        for table, columns in self._columns.items():
            for name, column in columns.items():
                if issubclass(column.py_type, self.__class__):
                    tables.append((table, name))
        return tables

    def delete(self, parent=None, rowid=None):
        """Recursively delete this instance's row and its foreign objects from the database. Before
           that, check if it is still referenced somewhere else.
        """
        # Before deleting this object from the database, we check if it is referenced in any
        # other table. This check is done differently depending on whether delete() was called
        # directly on this object or not.
        # If it was called directly we fail with a ForeignKeyError if the object is still
        # referenced somewhere else. In the other case, we do not raise a ForeignKeyError, but we
        # also do not delete this object.
        for table, name in self._find_references():
            where = f"{quote(name)} = ?"
            values = [self.rowid]
            ignore = parent is not None

            if table == parent:
                # Do not count the parent row that is about to be deleted.
                where = f"rowid != ? and " + where
                values.insert(0, rowid)

            row = self.execute(f"select count(*) from {quote(table)} where {where}", values).fetchone()
            if row[0] > 0:
                if ignore:
                    return
                else:
                    raise ForeignKeyError(f"cannot delete {self._table_name()}, "\
                            f"it is still referenced by {table}.{name}")

        table = self._table_name()
        columns = self._columns[table]

        for name, column in columns.items():
            if issubclass(column.py_type, Model):
                # Also delete referenced foreign key objects.
                reference = getattr(self, name)
                reference.delete(parent=table, rowid=self.rowid)

        self.execute(f"delete from {quote(table)} where rowid = ?", (self.rowid,))
        self.conn.commit()

        del self.rowid

    #
    # Low-level class methods.
    #
    @classmethod
    def _table_name(cls):
        """Return the sql table name for this class.
        """
        return cls.__name__.lower()

    @classmethod
    def _from_db(cls, row):
        """Convert a database table row into an instance of this class.
        """
        row = dict(row)
        if "rowid" not in row:
            raise ValueError(f"missing rowid column in row {row}")
        obj = cls.__new__(cls, **row)
        obj.rowid = row.pop("rowid")
        obj._initialize((), row, from_db=True)
        return obj

    #
    # Query methods.
    #
    @classmethod
    def all(cls):
        """Return an iterator over all objects.
        """
        return Query(cls)

    @classmethod
    def filter(cls, **selectors):
        """Return an iterator yielding all objects that match the <selectors>.
        """
        return cls.all().filter(**selectors)

    @classmethod
    def exclude(cls, **selectors):
        """Return an iterator excluding all objects that match the <selectors>.
        """
        return cls.all().exclude(**selectors)

    @classmethod
    def order_by(cls, *fields):
        """Return all objects ordered by <fields>.
        """
        return cls.all().order_by(*fields)

    @classmethod
    def range(cls, offset, limit):
        """Return <limit> number of objects starting at <offset>.
        """
        return cls.all().range(offset, limit)

    @classmethod
    def get(cls, **selectors):
        """Return a single object that matches the <selectors>.
        """
        for obj in cls.filter(**selectors):
            return obj
        else:
            raise KeyError("no object found for " + \
                           ", ".join(f"{k}={v!r}" for k, v in selectors.items()))

    @classmethod
    def select(cls, where, values=None, join=None):
        """Return an iterator yielding all objects that match the <where> clause.
           If the where clause references other tables, these must be specified
           with <join>.
        """
        return SQLQuery(cls, where, values if values is not None else [],
                        join if join is not None else [])


class _BaseQuery:
    """The base class for SQLQuery and Query classes.
    """

    def __init__(self, cls):
        self._cls = cls

        self._table = self._cls._table_name()
        self._columns = self._cls._columns[self._table]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.sql!r}>"

    def __iter__(self):
        """Execute the query, yielding objects from the parent class.
        """
        for row in self._cls.execute(self.sql, self._values):
            yield self._cls._from_db(row)

    def prepare_select_from(self):
        """Build the first part of the sql query: select ... from ... join ...
        """
        # Construct what to select, i.e. all columns from the table plus the rowid column.
        columns = list(self._columns) + ["rowid"]
        columns = ", ".join(f"{quote(self._table)}.{n}" for n in columns)

        # Add which table to select from, and with which other tables it must be joined to
        # fulfill the query.
        tables = quote(self._table)
        for t, n in sorted(self._join):
            tables += f" join {quote(n)} on {quote(self._table)}.{quote(t)} = {quote(n)}.rowid"

        return columns, tables

    def prepare_where(self):
        """Build the "where" part of the sql query.
        """
        raise NotImplementedError

    def prepare_order_by(self):
        """Build the "order by" part of the sql query.
        """
        raise NotImplementedError

    def prepare_limit(self):
        """Build the "limit" part of the sql query.
        """
        raise NotImplementedError

    def _translate_column_to_table(self, column):
        """Translate a column name to the name of the table it references.
        """
        return self._columns[column].py_type._table_name()

    def _translate_table_to_column(self, table):
        """Find the column name that references a table name.
        """
        for name, column in self._columns.items():
            if issubclass(column.py_type, Model) and column.py_type._table_name() == table:
                return name
        else:
            raise ValueError(f"there is no column that references {table}")

    @property
    def sql(self):
        """Build a complete sql statement from the different query arguments.
        """
        columns, tables = self.prepare_select_from()
        where = self.prepare_where()
        order_by = self.prepare_order_by()
        limit = self.prepare_limit()
        return f"select {columns} from {tables}{where}{order_by}{limit}"


class SQLQuery(_BaseQuery):
    """A low level query class that allows a more direct access to the underlying sql.
    """

    def __init__(self, cls, where, values, join):
        super().__init__(cls)
        self._where = where
        self._values = values
        self._join = set((self._translate_table_to_column(j), j) for j in join)

    def prepare_where(self):
        return f" where {self._where}"

    def prepare_order_by(self):
        return ""

    def prepare_limit(self):
        return ""


class Query(_BaseQuery):

    def __init__(self, cls):
        super().__init__(cls)

        self._join = set()
        self._where = []
        self._values = []
        self._order = []
        self._limit = None
        self._offset = None

    def filter(self, **selectors):
        """Add filtering expressions to the query.
        """
        join, where, values = self._translate_selector(selectors)
        self._join |= join
        self._where.append(" and ".join(where))
        self._values += values
        return self

    def exclude(self, **selectors):
        """Add exclusion expressions to the query.
        """
        join, where, values = self._translate_selector(selectors)
        self._join |= join
        self._where.append("not (" + " and ".join(where) + ")")
        self._values += values
        return self

    def order_by(self, *fields):
        """Add ordering expressions to the query.
        """
        for field in fields:
            direction = "desc" if field.startswith("-") else "asc"
            field = field.lstrip("-")
            column, join = self._translate_field(field)
            self._order.append(f"{column} {direction}")
            self._join |= join
        return self

    def range(self, offset, limit):
        """Add an offset and limit expression to the query.
        """
        if self._limit is not None:
            raise ValueError("range() can be used only once")
        self._offset = offset
        self._limit = limit
        return self

    def prepare_where(self):
        return " where " + " ".join(self._where) if self._where else ""

    def prepare_order_by(self):
        return " order by " + ", ".join(self._order) if self._order else ""

    def prepare_limit(self):
        if self._limit is not None:
            limit = f" limit {self._limit}"
        else:
            limit = ""

        if self._offset is not None:
            if self._limit is not None:
                limit += " "
            limit += f"offset {self._offset}"

        return limit

    #
    # Low level methods.
    #
    def _translate_field(self, field):
        """Translate an attribute name to a column name.
        """
        columns = field.split(".")
        if len(columns) == 1:
            return f"{quote(self._table)}.{quote(columns[0])}", set()
        elif len(columns) == 2:
            table = self._translate_column_to_table(columns[0])
            return f"{quote(table)}.{quote(columns[1])}", {(columns[0], table)}
        else:
            raise ValueError(f"invalid field {field!r}")

    def _translate_selector(self, selectors):
        """Translate django-style keyword argument selectors to sql expressions.
        """
        join = set()
        where = []
        values = []
        for key, value in selectors.items():
            parts = key.split("__")
            if len(parts) == 1:
                column = f"{quote(self._table)}.{quote(parts[0])}"
                if value is None:
                    operator = "is"
                else:
                    operator = "eq"
            elif len(parts) == 2:
                column, operator = parts
                column = f"{quote(self._table)}.{quote(column)}"
            elif len(parts) == 3:
                table, column, operator = parts
                newtable = self._translate_column_to_table(table)
                join.add((table, newtable))
                column = f"{quote(newtable)}.{quote(column)}"
            else:
                raise ValueError(f"invalid selector {key!r}")

            match operator:
                case "eq":
                    sql = f"{column} = ?"
                case "ne":
                    sql = f"{column} != ?"
                case "gt":
                    sql = f"{column} > ?"
                case "ge":
                    sql = f"{column} >= ?"
                case "lt":
                    sql = f"{column} < ?"
                case "le":
                    sql = f"{column} <= ?"
                case "is":
                    sql = f"{column} is ?"
                case "isnot":
                    sql = f"{column} is not ?"
                case "like":
                    sql = f"{column} like ?"
                case "contains":
                    sql = f"{column} like ?"
                    value = f"%{value.strip('%')}%"
                case "year":
                    sql = f"extract_datetime({column}, 'year') = ?"
                case _:
                    raise ValueError(f"invalid operator {operator!r}")

            where.append(sql)
            values.append(value)

        return join, where, values
