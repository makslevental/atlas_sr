import fire

class Calculator(object):
  """A simple calculator class."""
  test = 5
  def double(self, number):
    return 2 * number

if __name__ == '__main__':
  f: Calculator = fire.Fire(Calculator)
  print(f)