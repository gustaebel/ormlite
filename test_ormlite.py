import sqlite3
import unittest

from datetime import date

from ormlite import Model, UniqueError, ForeignKeyError


class Role(Model):

    name: (str, "pk")


class Department(Model):

    name: (str, "unique")
    city: (str, "unique")


class User(Model):

    name: str
    dept: Department
    role: Role
    created_at: (date, "null")
    active: bool
    data: (dict, "null")
    binary: (bytes, "null")


class Test(Model):

    dept: Department


class TestBasics(unittest.TestCase):

    def setUp(self):
        Model.connect(":memory:")

        try:
            self.d1 = Department("Department 1", "Austin")
            self.d2 = Department("Department 2", "Boston")
            self.d3 = Department("Department 3", "Chicago")
            r1 = Role("Team Lead")
            r2 = Role("Developer")
            r3 = Role("Intern")
            self.u1 = User("Alice", self.d2, r1, date(2020, 1, 1), True).save()
            self.u2 = User("Bob", self.d1, r2, date(2021, 1, 1), True).save()
            self.u3 = User("Charlie", self.d2, r2, None, True).save()
            self.u4 = User("David", self.d3, r3, date(2023, 1, 1), False, {"foo": 23},
                           "ÄÖÜäöüß".encode("utf-8")).save()
            self.t1 = Test(self.d3).save()
        except:
            Model.close()
            raise

    def tearDown(self):
        Model.close()

    def test_basic(self):
        self.assertEqual(len(list(Department.all())), 3)
        self.assertEqual(len(list(Role.all())), 3)
        self.assertEqual(len(list(User.all())), 4)

    def test_identity(self):
        self.assertNotEqual(self.u1.role.rowid, self.u2.role.rowid)
        self.assertEqual(self.u2.role.rowid, self.u3.role.rowid)

    def test_unique(self):
        self.assertRaises(UniqueError, User(name="Alice").save)
        self.assertRaises(UniqueError, Role(name="Developer").save)
        self.assertRaises(UniqueError, Department("Department 1", "Austin").save)
        try:
            Department("Department 1", "Dallas").save()
        except UniqueError:
            self.fail("this UniqueError should not have been raised")

    def test_delete(self):
        self.u1.delete()
        self.assertRaises(KeyError, User.get, name="Alice")
        self.assertRaises(KeyError, Role.get, name="Team Lead")
        self.assertEqual(Department.get(name="Department 2").rowid, self.d2.rowid)

    def test_delete_referenced1(self):
        self.assertRaises(ForeignKeyError, self.d1.delete)

    def test_delete_referenced2(self):
        self.assertRaises(ForeignKeyError, self.d3.delete)

    def test_delete_all(self):
        self.u1.delete()
        self.u2.delete()
        self.u3.delete()
        self.u4.delete()
        self.t1.delete()
        self.assertEqual(len(list(Department.all())), 0)
        self.assertEqual(len(list(Role.all())), 0)
        self.assertEqual(len(list(User.all())), 0)

    def test_types(self):
        self.assertIsInstance(self.u1.role, Role)
        self.assertIsInstance(self.u1.created_at, date)

    def test_foreign_key(self):
        self.assertEqual(User.get(dept__name__eq="Department 1").rowid, self.u2.rowid)

    def test_date(self):
        self.assertEqual(User.get(name="David").created_at, date(2023, 1, 1))

        users = [user.name for user in User.filter(created_at__year=2023)]
        self.assertEqual(users, ["David"])

        users = [user.name for user in User.filter(created_at__ge=date(2021, 1, 1))]
        self.assertEqual(users, ["Bob", "David"])

    def test_update(self):
        self.u1.name = "Anna"
        self.u1.update()
        self.assertEqual(self.u1.rowid, User.get(name="Anna").rowid)

        self.u1.role.name = "Management"
        self.u1.update()

        user = User.get(name="Anna")
        self.assertEqual(user.role.name, "Management")

        roles = {role.name for role in Role.all()}
        self.assertEqual(roles, {"Management", "Developer", "Intern"})

        user = User.get(name="Bob")
        user.role = Role("Engineer")
        user.update()
        roles = {role.name for role in Role.all()}
        self.assertEqual(roles, {"Management", "Developer", "Intern", "Engineer"})

    def test_get(self):
        user = User.get(name="Alice")
        self.assertEqual(user.rowid, self.u1.rowid)

        user = User.get(role__name__eq="Team Lead")
        self.assertEqual(user.rowid, self.u1.rowid)

    def test_get_null(self):
        user = User.get(created_at=None)
        self.assertEqual(user.rowid, self.u3.rowid)

        user = User.get(created_at__is=None)
        self.assertEqual(user.rowid, self.u3.rowid)

        users = [user.name for user in User.filter(created_at__isnot=None)]
        self.assertEqual(users, ["Alice", "Bob", "David"])

    def test_all(self):
        users = [user.name for user in User.all()]
        self.assertEqual(users, ["Alice", "Bob", "Charlie", "David"])

    def test_order(self):
        cities = [department.city for department in Department.order_by("city")]
        self.assertEqual(cities, ["Austin", "Boston", "Chicago"])

        cities = [department.city for department in Department.order_by("-city")]
        self.assertEqual(cities, ["Chicago", "Boston", "Austin"])

        cities = [user.dept.city for user in User.order_by("dept.city")]
        self.assertEqual(cities, ["Austin", "Boston", "Boston", "Chicago"])

    def test_exclude(self):
        cities = [department.city for department in Department.exclude(city__eq="Boston").order_by("city")]
        self.assertEqual(cities, ["Austin", "Chicago"])

        cities = [user.dept.city for user in User.exclude(dept__city__eq="Boston").order_by("dept.city")]
        self.assertEqual(cities, ["Austin", "Chicago"])

    def test_select(self):
        cities = [department.city for department in Department.select("city != 'Boston' order by city")]
        self.assertEqual(cities, ["Austin", "Chicago"])

        users = [user.name for user in User.select("department.city = 'Boston' order by user.name desc", join=["department"])]
        self.assertEqual(users, ["Charlie", "Alice"])

    def test_range(self):
        users = [user.name for user in User.range(0, 2)]
        self.assertEqual(users, ["Alice", "Bob"])

        users = [user.name for user in User.all().range(2, 4)]
        self.assertEqual(users, ["Charlie", "David"])

        self.assertRaises(ValueError, User.range(2, 4).range, 0, 1)

    def test_json(self):
        user = User.get(name="David")
        self.assertEqual(user.data, {"foo": 23})

        user.data["bar"] = 42
        user.update()
        user = User.get(name="David")
        self.assertEqual(user.data, {"foo": 23, "bar": 42})

    def test_bytes(self):
        user = User.get(name="David")
        self.assertEqual(user.binary, "ÄÖÜäöüß".encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
