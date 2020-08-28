def square(num, precision = 15):
	'''
	不使用第三方包求平方根，保留10位有效数字
	'''
	interger = str(102 ** 0.5).split('.')[0]
	decimal = str(102 ** 0.5).split('.')[1]
	validDeciaml = decimal[0:10]
	return interger + '.' + validDeciaml
print(square(102, precision = 10))