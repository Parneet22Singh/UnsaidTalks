#Take user input for name and age and print a greeting.
def Greeting(name,age):
  print(f"Hello {name}! Hope you are doing great!")
x=input("Enter your name:")
y=int(input("Enter your age:"))
Greeting(x,y)

#Check whether a number is +ve, -ve or 0
def check(x):
  if x>0:
    print("Positive")
  elif x<0:
    print("Negative")
  else:
    print("Positive")
x=int(input("Enter a number: "))
check(x)

#Perform all arithematic operations on 2 numbers.
def operations(a,b):
  print("1. Addition\n2.Subtraction\n3.Multiplication\n4.Division\n5.Modular Division\n6.Floor Division\n7.Exponent")
  ch=int(input("Enter your choice:"))
  if ch==1:
    return a+b
  elif ch==2:
    return a-b
  elif ch==3:
    return a*b
  elif ch==4:
    return a/b
  elif ch==5:
    return a//b
  elif ch==6:
    return a%b
  elif ch==7:
    return a**b
  else:
    print("Invalid choice")
x=int(input("Enter first number:"))
y=int(input("Enter second number:"))
z=operations(x,y)
print(f"The result of your desired operation is: {z}")

#WAP to swap values of 2 variables without using 3rd variable
#Python oriented
def swap(a,b):
  print(f"The numbers before swapping are:{a},{b}")
  a,b=b,a
  print(f"The numbers after swapping are:{a},{b}")
a=int(input("Enter the first number:"))
b=int(input("Enter the second number:"))
swap(a,b)
#General method to do the swap
def swap_alt():
  a=int(input("Enter the first number:"))
  b=int(input("Enter the second number:"))
  print(f"The numbers before swapping are:{a},{b}")
  a+=b
  b=a-b
  a=a-b
  print(f"The numbers after swapping are:{a},{b}")
swap_alt()
